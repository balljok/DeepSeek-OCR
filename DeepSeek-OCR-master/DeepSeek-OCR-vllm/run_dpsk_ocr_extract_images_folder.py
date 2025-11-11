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

from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from process.image_process import DeepseekOCRProcessor
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

from config import (
    MODEL_PATH,
    SKIP_REPEAT,
    MAX_CONCURRENCY,
    NUM_WORKERS,
    CROP_MODE,
)

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


def pdf_to_images_high_quality(
    pdf_path: str, dpi: int, image_format: str = "PNG"
) -> List[Image.Image]:
    """
    Convert all pages of a PDF document to high-quality PIL Images.

    Renders each page of the PDF at the specified DPI and converts them to
    PIL Image objects. Handles both PNG and other formats, with automatic
    conversion of RGBA/LA images to RGB with white background.

    Args:
        pdf_path: Path to the PDF file to convert.
        dpi: Resolution in dots per inch for rendering (e.g., 144, 300).
        image_format: Output image format (default: "PNG").

    Returns:
        List[Image.Image]: A list of PIL Images, one per PDF page, in order.

    Example:
        >>> images = pdf_to_images_high_quality("document.pdf", dpi=300)
        >>> print(f"Converted {len(images)} pages")
    """
    images = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            if image_format.upper() == "PNG":
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
            else:
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(
                        img, mask=img.split()[-1] if img.mode == "RGBA" else None
                    )
                    img = background

            images.append(img)

    return images


def process_pdf_pages(pdf_path: Path, dpi: int) -> Iterable[PageContext]:
    """Yield per-page contexts for a PDF rendered at the given DPI."""

    images = pdf_to_images_high_quality(str(pdf_path), dpi=dpi)

    for page_number, image in enumerate(images, start=1):
        yield PageContext(page_number=page_number, image=image)


def process_page(
    image: Image.Image, prompt: str, processor: DeepseekOCRProcessor
) -> DeepSeekRequest:
    """
    Prepare a single page image for OCR processing.

    Tokenizes the image using DeepseekOCRProcessor and packages it with
    the prompt into a format suitable for batch processing by the LLM.

    Args:
        image: PIL Image of the page to process.
        prompt: Text prompt to guide the OCR model's behavior.

    Returns:
        DeepSeekRequest: A cache item containing:
            - "prompt": The text prompt
            - "multi_modal_data": Dictionary with tokenized image data

    Note:
        Uses CROP_MODE from config for image preprocessing.
    """
    cache_item: DeepSeekRequest = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                conversation=prompt, images=[image], bos=True, eos=True, cropping=CROP_MODE
            )
        },
    }
    return cache_item


def reference_match(
    text: str,
) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    """
    Extract and categorize reference tags from OCR output text.

    Parses the OCR output to find references marked with special tags:
    <|ref|>...<|/ref|><|det|>...<|/det|>

    Separates image references from other types of references.

    Args:
        text: OCR output text containing reference tags.

    Returns:
        Tuple[List[Tuple[str, str, str]], List[str], List[str]]: A tuple containing:
            - matches: All reference matches found
            - matches_image: References specifically for images
            - matches_other: All other non-image references

    Example:
        >>> matches, images, others = reference_match(ocr_text)
        >>> print(f"Found {len(images)} image references")
    """
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches: List[Tuple[str, str, str]] = re.findall(pattern, text, re.DOTALL)

    matches_image: List[str] = []
    matches_other: List[str] = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_caption_text(content: str, caption_ref_token: str) -> str:
    """
    Extract the text content that follows an image_caption reference.

    Args:
        content: The full page OCR content.
        caption_ref_token: The caption reference token to search for.

    Returns:
        The caption text, or empty string if not found.
    """
    # Find the position of the caption reference
    pos = content.find(caption_ref_token)
    if pos == -1:
        return ""

    # Move past the reference token
    start = pos + len(caption_ref_token)

    # Find the next reference or double newline to determine caption end
    next_ref_pattern = r"<\|ref\|>"
    remaining = content[start:]

    # Look for the next reference tag
    next_ref_match = re.search(next_ref_pattern, remaining)

    # Look for paragraph break
    para_break_match = re.search(r"\n\s*\n", remaining)

    # Use the earlier boundary
    end_pos = len(remaining)
    if next_ref_match and para_break_match:
        end_pos = min(next_ref_match.start(), para_break_match.start())
    elif next_ref_match:
        end_pos = next_ref_match.start()
    elif para_break_match:
        end_pos = para_break_match.start()

    caption_text = remaining[:end_pos].strip()

    # Remove common HTML-like tags if present
    caption_text = re.sub(r"<[^>]+>", "", caption_text)

    return caption_text


def extract_coordinates_and_label(
    ref_text: Tuple[Any, ...], image_width: int, image_height: int
) -> Optional[Tuple[str, List[Any]]]:
    """
    Extract label type and coordinate points from a reference text tuple.

    Parses the reference tuple to extract the label type (e.g., "image")
    and the list of coordinate points. Coordinates are expected to be
    in a format that can be evaluated as a Python literal.

    Args:
        ref_text: Tuple containing reference information (label, coordinates).
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.

    Returns:
        Optional[Tuple[str, List]]: A tuple containing:
            - label_type: The type of reference (e.g., "image")
            - cor_list: List of coordinate points
        Returns None if extraction fails.

    Note:
        Uses ast.literal_eval to parse coordinates and logs failures.
    """
    try:
        label_type = ref_text[1]
        cor_list = ast.literal_eval(ref_text[2])
    except (ValueError, SyntaxError) as exc:
        logger.warning("Failed to parse coordinates for label %s: %s", ref_text, exc)
        return None

    return (label_type, cor_list)


def extract_images_from_page(
    image: Image.Image,
    refs: List[Tuple[str, str, str]],
    images_dir_path: Path,
    base_name: str,
    page_num: int,
    dpi: int,
    captions: Optional[List[Tuple[List[List[int]], str]]] = None,
) -> int:
    """
    Extract and save embedded images from a PDF page based on detected references.

    Processes reference coordinates to crop and extract images from the page.
    Coordinates are normalized (0-999 scale) and converted to pixel coordinates
    based on the actual image dimensions. Extracted images are saved in a
    images subdirectory.

    Args:
        image: PIL Image of the PDF page.
        refs: List of reference tuples from reference_match containing
            the full match, label, and coordinate payload.
        images_dir_path: Directory where extracted images should be stored.
        base_name: Base filename for extracted images (usually PDF basename).
        page_num: Current page number (1-indexed).
        dpi: DPI to use when saving extracted images.
        captions: Optional list of (coordinates, caption_text) tuples for image captions.

    Returns:
        int: Number of images successfully extracted from the page.

    Note:
        - Creates the images subdirectory if it doesn't exist
        - Filenames format: {base_name}_{page:03d}_{image:02d}.png
        - Logs warnings for invalid coordinates or extraction errors
        - Embeds caption text in PNG Title metadata when available
    """
    image_width, image_height = image.size
    img_idx = 0

    # Create the images subfolder
    images_dir_path.mkdir(parents=True, exist_ok=True)

    # Convert caption coordinates to pixel space and track which captions have been used
    caption_boxes: List[Tuple[Tuple[int, int, int, int], str]] = []
    if captions:
        for caption_coords, caption_text in captions:
            for coords in caption_coords:
                if len(coords) == 4:
                    cx1, cy1, cx2, cy2 = coords
                    # Convert to pixel coordinates
                    px1 = int(cx1 / 999 * image_width)
                    py1 = int(cy1 / 999 * image_height)
                    px2 = int(cx2 / 999 * image_width)
                    py2 = int(cy2 / 999 * image_height)
                    caption_boxes.append(((px1, py1, px2, py2), caption_text))

    used_captions = set()

    def find_matching_caption(
        img_x1: int, img_y1: int, img_x2: int, img_y2: int
    ) -> Optional[str]:
        """Find a caption that appears below the image."""
        best_caption = None
        best_distance = float("inf")
        best_idx = -1

        # Tolerance for horizontal alignment (10% of image width)
        h_tolerance = int(image_width * 0.1)
        # Max vertical gap to consider (5% of image height)
        max_v_gap = int(image_height * 0.05)

        for idx, ((cx1, cy1, cx2, cy2), caption_text) in enumerate(caption_boxes):
            if idx in used_captions:
                continue

            # Check if caption is below the image
            if cy1 < img_y2:
                continue

            # Check vertical proximity
            v_gap = cy1 - img_y2
            if v_gap > max_v_gap:
                continue

            # Check horizontal alignment (caption should overlap or be near image horizontally)
            h_overlap = min(img_x2, cx2) - max(img_x1, cx1)
            if h_overlap < -h_tolerance:
                continue

            # This caption is a candidate; prefer the closest one
            if v_gap < best_distance:
                best_distance = v_gap
                best_caption = caption_text
                best_idx = idx

        if best_idx >= 0:
            used_captions.add(best_idx)

        return best_caption

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                if label_type == "image":
                    for points in points_list:
                        x1, y1, x2, y2 = points

                        # Convert normalized coordinates (0-999) to pixel coordinates
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)

                        try:
                            # Crop and save the image
                            cropped = image.crop((x1, y1, x2, y2))

                            # Generate filename: basename_pageXX_imageYY.png
                            img_filename = (
                                f"{base_name}_{page_num:03d}_{(img_idx+1):02d}.png"
                            )
                            img_path = images_dir_path / img_filename

                            # Check for matching caption
                            caption_text = find_matching_caption(x1, y1, x2, y2)

                            if caption_text:
                                # Create PNG metadata with caption in Title field
                                metadata = PngImagePlugin.PngInfo()
                                metadata.add_text("Title", caption_text)
                                cropped.save(
                                    img_path, "PNG", dpi=(dpi, dpi), pnginfo=metadata
                                )
                                logger.debug(
                                    "Saved image %s with caption: %s",
                                    img_filename,
                                    (
                                        caption_text[:50] + "..."
                                        if len(caption_text) > 50
                                        else caption_text
                                    ),
                                )
                            else:
                                cropped.save(img_path, "PNG", dpi=(dpi, dpi))

                            img_idx += 1

                        except Exception as exc:
                            logger.warning(
                                "Failed to extract image for %s (page %d): %s",
                                base_name,
                                page_num,
                                exc,
                            )
        except Exception as exc:
            logger.warning(
                "Failed to process reference on page %d for %s: %s",
                page_num,
                base_name,
                exc,
            )

    return img_idx


def process_document(
    pdf_path: Path,
    prompt: str,
    dpi: int,
    images_dir: str,
    force_overwrite: bool = False,
) -> ProcessResult:
    """
    Process a single PDF document with DeepSeek-OCR.

    Converts the PDF to images, runs OCR on each page, extracts embedded images,
    and saves the results as markdown files. Supports skipping already processed
    documents unless force_overwrite is enabled.

    Args:
        pdf_path: Path to the PDF file to process.
        prompt: OCR prompt to guide the model's behavior.
        dpi: Resolution for rendering PDF pages and saving extracted images.
        images_dir: Relative directory name to store extracted images.
        force_overwrite: If False, skip processing if output directory exists.

    Returns:
        ProcessResult: Result object containing status, output directory, and
        any raised exception. Status values:

            - "success": Document processed successfully
            - "skipped": Output exists and force_overwrite is False
            - "error": Processing failed with an exception

    Output Files:
        - {basename}_dpsk/{basename}_det.mmd: Raw OCR output with references
        - {basename}_dpsk/{basename}.mmd: Cleaned markdown with image links
        - {basename}_dpsk/{images_dir}/*.png: Extracted images from the PDF

    Example:
    >>> result = process_document(
    ...     Path("report.pdf"),
    ...     prompt="<image>\n<|grounding|>Convert to markdown.",
    ...     dpi=144,
    ...     images_dir="bilder",
    ...     force_overwrite=False,
    ... )
    >>> print(result.status)
    """
    output_dir: Optional[Path] = None

    try:
        pdf_basename = pdf_path.stem
        output_dir = pdf_path.parent / f"{pdf_basename}_dpsk"

        # Check if output directory exists and skip if force_overwrite is False
        if output_dir.exists() and not force_overwrite:
            logger.info("Skipping %s (output exists)", pdf_path.name)
            return ProcessResult(status="skipped", output_dir=output_dir)

        # Get model instance (lazy initialization)
        llm, sampling_params = _initialize_model()

        output_dir.mkdir(exist_ok=True)

        images_dir_path = output_dir / images_dir

        page_contexts = list(process_pdf_pages(pdf_path, dpi))

        if not page_contexts:
            logger.warning("No pages extracted from %s", pdf_path.name)
            return ProcessResult(status="success", output_dir=output_dir)

        processor = DeepseekOCRProcessor()
        process_single = partial(process_page, prompt=prompt, processor=processor)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            requests: List[DeepSeekRequest] = list(
                tqdm(
                    executor.map(
                        process_single, (context.image for context in page_contexts)
                    ),
                    total=len(page_contexts),
                    desc=f"Pre-processing {pdf_path.name}",
                    leave=True,
                )
            )

        outputs_list = llm.generate(  # type: ignore[arg-type]
            cast(List[Any], requests), sampling_params=sampling_params
        )

        contents_det = ""
        contents = ""
        total_images_extracted = 0

        for output, context in tqdm(
            zip(outputs_list, page_contexts),
            total=len(page_contexts),
            desc="Processing pages",
            leave=True,
        ):
            content = output.outputs[0].text

            # Check for completion
            if "<｜end▁of▁sentence｜>" in content:
                content = content.replace("<｜end▁of▁sentence｜>", "")
            else:
                if SKIP_REPEAT:
                    continue

            page_num = context.page_number
            page_marker = f"\n<--- Page {page_num} --->"
            contents_det += content + f"\n{page_marker}\n"

            # Extract references
            matches_ref, matches_images, matches_other = reference_match(content)
            context.references = (matches_ref, matches_images, matches_other)

            # Extract image captions
            captions: List[Tuple[List[List[int]], str]] = []
            for full_match, label, coords_str in matches_ref:
                if label == "image_caption":
                    try:
                        coords_list = ast.literal_eval(coords_str)
                        caption_text = extract_caption_text(content, full_match)
                        if caption_text:
                            captions.append((coords_list, caption_text))
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse caption on page %d: %s", page_num, exc
                        )

            context.captions = captions

            # Extract images from this page
            num_images = extract_images_from_page(
                context.image,
                matches_ref,
                images_dir_path,
                pdf_basename,
                page_num,
                dpi,
                captions=captions,
            )
            total_images_extracted += num_images

            # Replace image references in markdown
            for idx, a_match_image in enumerate(matches_images):
                img_filename = f"{pdf_basename}_{page_num:03d}_{(idx+1):02d}.png"
                image_markdown_path = (Path(images_dir) / img_filename).as_posix()
                content = content.replace(
                    a_match_image, f"![]({image_markdown_path})\n"
                )

            # Clean up other references
            for a_match_other in matches_other:
                content = (
                    content.replace(a_match_other, "")
                    .replace("\\coloneqq", ":=")
                    .replace("\\eqqcolon", "=:")
                    .replace("\n\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n")
                )

            context.content = content
            contents += content + f"\n{page_marker}\n"

        # Save output files
        mmd_det_path = output_dir / f"{pdf_basename}_det.mmd"
        mmd_path = output_dir / f"{pdf_basename}.mmd"

        with open(mmd_det_path, "w", encoding="utf-8") as f:
            f.write(contents_det)

        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(contents)

        logger.info("Processed %s", pdf_path.name)
        return ProcessResult(status="success", output_dir=output_dir)

    except Exception as exc:
        logger.exception("Failed to process %s", pdf_path)
        return ProcessResult(status="error", output_dir=output_dir, error=exc)


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Root folder containing PDFs to process",
)
@click.option(
    "-p",
    "--prompt",
    default="<image>\\n<|grounding|>Convert the document to markdown.",
    type=str,
    show_default=True,
    help="Prompt for document processing",
)
@click.option(
    "--dpi",
    default=144,
    type=int,
    show_default=True,
    help="DPI for image extraction",
)
@click.option(
    "--images-dir-name",
    default="bilder",
    show_default=True,
    type=str,
    help="Name of the subdirectory to store extracted images",
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
    dpi: int,
    images_dir_name: str,
    force_overwrite: bool,
) -> None:
    """
    Recursively process all PDFs in a folder with DeepSeek-OCR.

    Main entry point for batch PDF processing. Walks through the input directory
    tree, finds all PDF files, and processes each one with OCR. Provides progress
    tracking and summary statistics.

    Args:
        input: Root folder path containing PDFs to process.
        prompt: OCR prompt for guiding model behavior.
        dpi: Resolution for rendering pages and saving images.
        images_dir_name: Subfolder name for saving extracted images.
        force_overwrite: If True, reprocess PDFs even if output exists.

    Output:
        For each PDF, creates a {basename}_dpsk directory containing:
        - Markdown files with OCR results
        - Extracted images in the configured images subdirectory

    Statistics:
        Displays summary of PDFs found, processed, skipped, and errors.

    Cleanup:
        Ensures GPU resources are properly released via finally block.

    Example:
        Called via Click CLI:
    $ python script.py -i /path/to/pdfs --dpi 300 --images-dir-name scans
    $ python script.py -i /path/to/pdfs --force-overwrite
    """
    pdf_count = 0
    success_count = 0
    error_count = 0
    skipped_count = 0

    try:
        for root, _, files in os.walk(input):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    pdf_count += 1
                    try:
                        result = process_document(
                            Path(pdf_path),
                            prompt=prompt,
                            dpi=dpi,
                            images_dir=images_dir_name,
                            force_overwrite=force_overwrite,
                        )
                    except Exception:
                        error_count += 1
                        logger.exception("Unhandled error processing %s", pdf_path)
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
                                pdf_path,
                                result.error,
                            )

        print(
            f"\n✓ Processing complete: {pdf_count} PDF(s) found, "
            f"{success_count} processed, {skipped_count} skipped, {error_count} error(s)"
        )

    finally:
        # Ensure cleanup happens
        _cleanup_resources()


if __name__ == "__main__":
    main()
