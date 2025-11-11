#!/usr/bin/env python3
"""Extract images from PDFs using DeepSeek-OCR."""

import ast
import atexit
import gc
import logging
import os
import io
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, TypedDict, cast

import click
import fitz
import torch
from PIL import Image, PngImagePlugin
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# CUDA setup
if torch.version.cuda == "11.8":  # type: ignore[attr-defined]
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import (
    MODEL_PATH,
    PROMPT,
    SKIP_REPEAT,
    MAX_CONCURRENCY,
    NUM_WORKERS,
    CROP_MODE,
)

from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from process.image_process import DeepseekOCRProcessor
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Configure logging: suppress INFO from third-party libraries
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Suppress vLLM and other third-party INFO/DEBUG messages
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

Image.MAX_IMAGE_PIXELS = None


# Typed structures for better type safety
class DeepSeekRequest(TypedDict):
    prompt: str
    multi_modal_data: Dict[str, Any]


@dataclass
class ProcessResult:
    status: Literal["success", "skipped", "error"]
    output_dir: Optional[Path] = None
    error: Optional[Exception] = None


@dataclass
class PageContext:
    """Lightweight container for per-page processing artifacts."""

    page_number: int
    image: Image.Image
    references: Optional[Tuple[List[Tuple[str, str, str]], List[str], List[str]]] = None
    content: str = ""
    captions: List[Tuple[List[List[int]], str]] = field(default_factory=list)


# Global model instance (lazy-loaded)
_DEEPSEEK_LLM: Optional[LLM] = None
_SAMPLING_PARAMS: Optional[SamplingParams] = None


def _cleanup_resources() -> None:
    """
    Clean up GPU resources and distributed process groups on exit.

    This function ensures proper cleanup of:
    - DeepSeek-OCR model instance
    - PyTorch distributed process groups
    - CUDA memory caches
    - Python garbage collection

    Called automatically via atexit when the program terminates.
    """
    global _DEEPSEEK_LLM
    if _DEEPSEEK_LLM is not None:
        logger.debug("Clearing DeepSeek LLM instance")
        del _DEEPSEEK_LLM
        _DEEPSEEK_LLM = None

    if torch.distributed.is_initialized():
        logger.debug("Destroying torch distributed process group")
        torch.distributed.destroy_process_group()

    if torch.cuda.is_available():
        logger.debug("Releasing CUDA cache")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def _initialize_model() -> tuple[LLM, SamplingParams]:
    """
    Initialize the DeepSeek-OCR model with lazy loading pattern.

    Creates and configures the LLM model only when first called, then returns
    the cached instance on subsequent calls. This pattern optimizes resource
    usage and ensures the model is only loaded when actually needed.

    Returns:
        tuple[LLM, SamplingParams]: A tuple containing:
            - LLM: The initialized DeepSeek-OCR model instance
            - SamplingParams: Configured sampling parameters for text generation

    Note:
        - Registers a cleanup handler on first initialization
        - Uses global variables for singleton pattern
        - Model configuration is loaded from config.py
    """
    global _DEEPSEEK_LLM, _SAMPLING_PARAMS

    if _DEEPSEEK_LLM is None:
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

        _DEEPSEEK_LLM = LLM(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=True,
            max_model_len=4096,  # 8196
            swap_space=0,
            max_num_seqs=MAX_CONCURRENCY,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.98,  # 0.9
            disable_mm_preprocessor_cache=True,
        )

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
            )
        ]

        _SAMPLING_PARAMS = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            logits_processors=logits_processors,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        # Register cleanup handler
        atexit.register(_cleanup_resources)

    assert _DEEPSEEK_LLM is not None
    if _SAMPLING_PARAMS is None:
        raise RuntimeError("Sampling parameters failed to initialize")

    return _DEEPSEEK_LLM, _SAMPLING_PARAMS

def process_image(
    image_path: Path,
    prompt: str,
    force_overwrite: bool,
) -> ProcessResult:
    """
    Process a single image with DeepSeek-OCR and save result as PNG metadata.

    Args:
        image_path: Path to the input PNG image
        prompt: Text prompt to use for OCR processing
        force_overwrite: If True, process even if Description metadata exists

    Returns:
        ProcessResult indicating success, skip, or error status
    """
    try:
        # Load image and check existing metadata
        image = Image.open(image_path)
        
        if image is not None:
            image = image.convert("RGB")

            # Check if Description already exists
            existing_description = image.info.get("Description", "")
            if existing_description and not force_overwrite:
                logger.debug("Skipping %s - Description metadata already exists", image_path)
                return ProcessResult(status="skipped")

            # Initialize model
            llm, sampling_params = _initialize_model()
            
            # Prepare processor
            processor = DeepseekOCRProcessor.from_pretrained(MODEL_PATH)

            # Create request using the same pattern as the PDF processor
            request: DeepSeekRequest = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": processor.tokenize_with_images(
                        conversation=prompt, images=[image], bos=True, eos=True, cropping=CROP_MODE
                    )
                },
            }

            # Run inference
            outputs = llm.generate([request], sampling_params)
            
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("Model returned no output")
            
            result_text = outputs[0].outputs[0].text

            # Save result as PNG metadata
            metadata = PngImagePlugin.PngInfo()
            
            # Preserve existing metadata except Description
            for key, value in image.info.items():
                if key != "Description":
                    metadata.add_text(key, str(value))
            
            # Add new Description
            metadata.add_text("Description", result_text)
            
            # Save image with updated metadata
            image.save(image_path, "PNG", pnginfo=metadata)
            
            logger.debug("Successfully processed %s", image_path)
            return ProcessResult(status="success", output_dir=image_path.parent)
        
    except Exception as e:
        logger.error("Error processing %s: %s", image_path, e)
        return ProcessResult(status="error", error=e)

    # Restore the original PROMPT
    config.PROMPT = original_prompt


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Root folder containing images to process",
)
@click.option(
    "-p",
    "--prompt",
    default="<image>\nDescribe this image in detail.",
    type=str,
    show_default=True,
    help="Prompt for image processing",
)
@click.option(
    "--images-dir-name",
    default="bilder",
    show_default=True,
    type=str,
    help="Name of the subdirectory where images to be processed are located",
)
@click.option(
    "--force-overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output directories (default: skip existing)",
)
def main(
    input: str,
    prompt: str,
    images_dir_name: str,
    force_overwrite: bool,
) -> None:
    image_count = 0
    success_count = 0
    error_count = 0
    skipped_count = 0

    try:
        for root, _, files in os.walk(input):
            for file in files:
                if file.lower().endswith(".png") and os.path.basename(os.path.dirname(os.path.join(root, file))) == images_dir_name:
                    image_path = os.path.join(root, file)
                    image_count += 1
                    try:
                        result = process_image(
                            Path(image_path),
                            prompt=prompt,
                            force_overwrite=force_overwrite,
                        )
                    except Exception:
                        error_count += 1
                        logger.exception("Unhandled error processing %s", image_path)
                        continue

                    if result.status == "success":
                        success_count += 1
                    elif result.status == "skipped":
                        skipped_count += 1
                    else:
                        error_count += 1
                        if result.error:
                            logger.error(
                                "Processing failed for %s: %s",
                                image_path,
                                result.error,
                            )

        print(
            f"\n✓ Processing complete: {image_count} image(s) found, "
            f"{success_count} processed, {skipped_count} skipped, {error_count} error(s)"
        )

    finally:
        # Ensure cleanup happens
        _cleanup_resources()

if __name__ == "__main__":
    main()