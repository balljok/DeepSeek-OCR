#!/usr/bin/env python3
"""
Batch process PDFs in a folder structure using DeepSeek-OCR.
Recursively processes all PDFs, extracts images, and saves results.
"""

import os
import sys
import argparse
import fitz  # PyMuPDF
import io
import re
from pathlib import Path
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# CUDA setup
if torch.version.cuda == "11.8":
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
import config  # Import config module to temporarily modify PROMPT

from PIL import Image, ImageOps
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor


class Colors:
    """ANSI color codes for terminal output"""

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def pdf_to_images_high_quality(
    pdf_path: str, dpi: int = 144, image_format: str = "PNG"
) -> List[Image.Image]:
    """
    Convert PDF pages to high-quality images.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for conversion (default 144)
        image_format: Output format (default PNG)

    Returns:
        List of PIL Images
    """
    images = []
    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

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

    pdf_document.close()
    return images


def re_match(text: str) -> Tuple[List, List, List]:
    """
    Extract reference tags from OCR output.

    Args:
        text: OCR output text

    Returns:
        Tuple of (all_matches, image_matches, other_matches)
    """
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])

    return matches, matches_image, matches_other


def extract_coordinates_and_label(
    ref_text: Tuple, image_width: int, image_height: int
) -> Tuple[str, List]:
    """
    Extract label type and coordinates from reference text.

    Args:
        ref_text: Tuple containing reference information
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        Tuple of (label_type, coordinates_list)
    """
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"{Colors.RED}Error extracting coordinates: {e}{Colors.RESET}")
        return None

    return (label_type, cor_list)


def extract_images_from_page(
    image: Image.Image,
    refs: List,
    output_dir: Path,
    base_name: str,
    page_num: int,
    dpi: int = 300,
) -> int:
    """
    Extract images from a page based on detected coordinates.

    Args:
        image: PIL Image of the page
        refs: List of reference matches
        output_dir: Directory to save extracted images
        base_name: Base name for output files
        page_num: Page number
        dpi: DPI for extraction (default 300)

    Returns:
        Number of images extracted
    """
    image_width, image_height = image.size
    img_idx = 0

    # Create the 'bilder' subfolder
    bilder_dir = output_dir / "bilder"
    bilder_dir.mkdir(exist_ok=True)

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
                            img_path = bilder_dir / img_filename

                            # Save at 300 DPI
                            cropped.save(img_path, "PNG", dpi=(dpi, dpi))
                            img_idx += 1

                        except Exception as e:
                            print(
                                f"{Colors.YELLOW}Warning: Failed to extract image {img_idx}: {e}{Colors.RESET}"
                            )
        except Exception as e:
            continue

    return img_idx


def load_image(image_path: Path) -> Image.Image:
    """
    Load an image and correct its orientation based on EXIF data.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image with corrected orientation
    """
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"{Colors.YELLOW}Warning loading image {image_path}: {e}{Colors.RESET}")
        try:
            return Image.open(image_path)
        except:
            return None


def process_single_image(image: Image.Image, prompt: str) -> dict:
    """
    Prepare a single image for OCR processing.

    Args:
        image: PIL Image to process
        prompt: Prompt for OCR

    Returns:
        Dictionary with prompt and image data
    """
    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=CROP_MODE
            )
        },
    }
    return cache_item


def process_pdf(
    pdf_path: Path,
    llm: LLM,
    sampling_params: SamplingParams,
    prompt: str,
    output_dpi: int = 300,
) -> bool:
    """
    Process a single PDF file with DeepSeek-OCR.

    Args:
        pdf_path: Path to PDF file
        llm: Initialized LLM model
        sampling_params: Sampling parameters for generation
        prompt: OCR prompt
        output_dpi: DPI for extracted images

    Returns:
        True if successful, False otherwise
    """
    try:
        pdf_basename = pdf_path.stem
        output_dir = pdf_path.parent / f"{pdf_basename}_dpsk"
        output_dir.mkdir(exist_ok=True)

        print(f"{Colors.CYAN}Processing: {pdf_path.name}{Colors.RESET}")
        print(f"{Colors.BLUE}Output directory: {output_dir}{Colors.RESET}")

        # Convert PDF to images
        print(f"{Colors.YELLOW}Converting PDF to images...{Colors.RESET}")
        images = pdf_to_images_high_quality(str(pdf_path), dpi=144)

        # Prepare batch inputs
        print(f"{Colors.YELLOW}Preparing images for OCR...{Colors.RESET}")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(
                tqdm(
                    executor.map(lambda img: process_single_image(img, prompt), images),
                    total=len(images),
                    desc="Pre-processing images",
                    leave=False,
                )
            )

        # Run OCR
        print(f"{Colors.YELLOW}Running DeepSeek-OCR...{Colors.RESET}")
        outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

        # Process results
        print(f"{Colors.YELLOW}Processing results...{Colors.RESET}")
        contents_det = ""
        contents = ""
        total_images_extracted = 0

        for page_num, (output, img) in enumerate(
            tqdm(
                zip(outputs_list, images),
                total=len(images),
                desc="Processing pages",
                leave=False,
            ),
            start=1,
        ):
            content = output.outputs[0].text

            # Check for completion
            if "<｜end▁of▁sentence｜>" in content:
                content = content.replace("<｜end▁of▁sentence｜>", "")
            else:
                if SKIP_REPEAT:
                    continue

            page_marker = f"\n<--- Page {page_num} --->"
            contents_det += content + f"\n{page_marker}\n"

            # Extract references
            matches_ref, matches_images, matches_other = re_match(content)

            # Extract images from this page
            num_images = extract_images_from_page(
                img, matches_ref, output_dir, pdf_basename, page_num, output_dpi
            )
            total_images_extracted += num_images

            # Replace image references in markdown
            for idx, a_match_image in enumerate(matches_images):
                img_filename = f"{pdf_basename}_{page_num:03d}_{(idx+1):02d}.png"
                content = content.replace(
                    a_match_image, f"![](bilder/{img_filename})\n"
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

            contents += content + f"\n{page_marker}\n"

        # Save output files
        mmd_det_path = output_dir / f"{pdf_basename}_det.mmd"
        mmd_path = output_dir / f"{pdf_basename}.mmd"

        with open(mmd_det_path, "w", encoding="utf-8") as f:
            f.write(contents_det)

        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(contents)

        print(f"{Colors.GREEN}✓ Successfully processed {pdf_path.name}{Colors.RESET}")
        print(f"{Colors.GREEN}  - Pages: {len(images)}{Colors.RESET}")
        print(
            f"{Colors.GREEN}  - Images extracted: {total_images_extracted}{Colors.RESET}"
        )
        print(f"{Colors.GREEN}  - Output: {output_dir}{Colors.RESET}")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Failed to process {pdf_path.name}: {e}{Colors.RESET}")
        return False


def process_extracted_images(
    output_dir: Path,
    llm: LLM,
    sampling_params: SamplingParams,
) -> int:
    """
    Process all extracted images in the bilder folder with Free OCR.
    Based on run_dpsk_ocr_image.py implementation.

    Args:
        output_dir: Directory containing the bilder folder
        llm: Initialized LLM model
        sampling_params: Sampling parameters for generation

    Returns:
        Number of images processed
    """
    bilder_dir = output_dir / "bilder"

    if not bilder_dir.exists():
        return 0

    # Find all PNG images in bilder folder
    image_files = sorted(bilder_dir.glob("*.png"))

    if not image_files:
        return 0

    print(
        f"{Colors.YELLOW}Processing {len(image_files)} extracted image(s) with Free OCR...{Colors.RESET}"
    )

    # Prepare Free OCR prompt - matching run_dpsk_ocr_image.py
    free_ocr_prompt = "<image>\nDescribe this image in general."

    # Save the original PROMPT and temporarily replace it
    original_prompt = config.PROMPT
    config.PROMPT = free_ocr_prompt

    # Load and prepare images - matching run_dpsk_ocr_image.py workflow
    batch_inputs = []
    valid_images = []

    for img_path in tqdm(image_files, desc="Loading images", leave=False):
        try:
            # Load image with EXIF correction (same as run_dpsk_ocr_image.py)
            image = load_image(img_path)
            if image is not None:
                image = image.convert("RGB")

                # Tokenize with images (same as run_dpsk_ocr_image.py line 214)
                if "<image>" in free_ocr_prompt:
                    image_features = DeepseekOCRProcessor().tokenize_with_images(
                        images=[image], bos=True, eos=True, cropping=CROP_MODE
                    )
                else:
                    image_features = ""

                # Build request matching run_dpsk_ocr_image.py line 176-179
                cache_item = {
                    "prompt": free_ocr_prompt,
                    "multi_modal_data": {"image": image_features},
                }
                batch_inputs.append(cache_item)
                valid_images.append(img_path)
        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning: Failed to load {img_path.name}: {e}{Colors.RESET}"
            )

    # Restore the original PROMPT
    config.PROMPT = original_prompt

    if not batch_inputs:
        return 0

    # Run OCR on all images
    print(f"{Colors.YELLOW}Running Free OCR on images...{Colors.RESET}")
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    # Save results - matching run_dpsk_ocr_image.py save logic (lines 223-240)
    processed_count = 0
    for img_path, output in tqdm(
        zip(valid_images, outputs_list),
        total=len(valid_images),
        desc="Saving OCR results",
        leave=False,
    ):
        try:
            content = output.outputs[0].text
            content = content.replace("<｜end▁of▁sentence｜>", "")

            # Save original output first
            mmd_ori_path = img_path.with_suffix(".ori.mmd")
            with open(mmd_ori_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Process output (matching run_dpsk_ocr_image.py lines 228-239)
            if "<image>" in free_ocr_prompt:
                outputs = content

                # Extract and remove grounding references
                matches_ref, matches_images, mathes_other = re_match(outputs)

                # Remove image references (they don't apply to standalone images)
                for idx, a_match_image in enumerate(matches_images):
                    outputs = outputs.replace(a_match_image, "")

                # Remove other references and clean up
                for idx, a_match_other in enumerate(mathes_other):
                    outputs = (
                        outputs.replace(a_match_other, "")
                        .replace("\\coloneqq", ":=")
                        .replace("\\eqqcolon", "=:")
                    )

                # Save cleaned markdown file with same basename as image
                mmd_path = img_path.with_suffix(".mmd")
                with open(mmd_path, "w", encoding="utf-8") as f:
                    f.write(outputs)

            processed_count += 1

        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning: Failed to save OCR for {img_path.name}: {e}{Colors.RESET}"
            )

    return processed_count


def translate_with_llama3(
    output_dir: Path,
    llama_model_path: str = "Qwen/Qwen2.5-3B-Instruct",
) -> int:
    """
    Translate image descriptions to Swedish using an LLM.

    Args:
        output_dir: Directory containing the bilder folder
        llama_model_path: Path to LLM model (default: Qwen2.5-3B-Instruct - free and open)

    Returns:
        Number of files translated
    """
    bilder_dir = output_dir / "bilder"

    if not bilder_dir.exists():
        return 0

    # Find all .mmd files (not .ori.mmd)
    mmd_files = sorted(
        [f for f in bilder_dir.glob("*.mmd") if not f.name.endswith(".ori.mmd")]
    )

    if not mmd_files:
        return 0

    print(
        f"{Colors.YELLOW}Translating {len(mmd_files)} description(s) to Swedish...{Colors.RESET}"
    )

    # Initialize translation model
    try:
        llama_llm = LLM(
            model=llama_model_path,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.98,
            dtype="float16",
        )

        llama_sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
        )
    except Exception as e:
        print(f"{Colors.RED}Failed to load translation model: {e}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Tip: Use an open model like 'Qwen/Qwen2.5-3B-Instruct' or 'microsoft/Phi-3-mini-4k-instruct'{Colors.RESET}"
        )
        return 0

    # Prepare batch inputs
    batch_prompts = []
    valid_files = []

    for mmd_path in tqdm(mmd_files, desc="Loading descriptions", leave=False):
        try:
            with open(mmd_path, "r", encoding="utf-8") as f:
                description = f.read().strip()

            if description:
                # Create prompt for Llama3
                prompt = f"""{description}. Shorten the text and translate the it to Swedish."""

                batch_prompts.append(prompt)
                valid_files.append(mmd_path)
        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning: Failed to load {mmd_path.name}: {e}{Colors.RESET}"
            )

    if not batch_prompts:
        return 0

    # Run Llama3 translation
    print(f"{Colors.YELLOW}Running Llama3 translation...{Colors.RESET}")
    outputs_list = llama_llm.generate(batch_prompts, llama_sampling_params)

    # Save results
    translated_count = 0
    for mmd_path, output in tqdm(
        zip(valid_files, outputs_list),
        total=len(valid_files),
        desc="Saving translations",
        leave=False,
    ):
        try:
            translation = output.outputs[0].text.strip()

            # Save Swedish translation with _sv suffix
            sv_path = mmd_path.with_suffix(".sv.mmd")
            with open(sv_path, "w", encoding="utf-8") as f:
                f.write(translation)

            translated_count += 1

        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning: Failed to save translation for {mmd_path.name}: {e}{Colors.RESET}"
            )

    return translated_count


def find_pdfs(root_path: Path) -> List[Path]:
    """
    Recursively find all PDF files in a directory.

    Args:
        root_path: Root directory to search

    Returns:
        List of PDF file paths
    """
    pdf_files = []
    for pdf_path in root_path.rglob("*.pdf"):
        # Skip PDFs in output directories (those ending with _dpsk)
        if "_dpsk" not in str(pdf_path):
            pdf_files.append(pdf_path)
    return sorted(pdf_files)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Batch process PDFs with DeepSeek-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/pdfs
  %(prog)s -i /path/to/pdfs --dpi 300
  %(prog)s -i /path/to/pdfs --prompt "<image>\\n<|grounding|>Convert the document to markdown."
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Root folder containing PDFs to process",
    )

    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for extracted images (default: 300)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=PROMPT,
        help="OCR prompt (default: from config.py)",
    )

    parser.add_argument(
        "--llama-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="LLM model for translation (default: Qwen/Qwen2.5-3B-Instruct - free and open)",
    )

    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip the third pass (translation to Swedish)",
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(
            f"{Colors.RED}Error: Input path does not exist: {input_path}{Colors.RESET}"
        )
        sys.exit(1)

    if not input_path.is_dir():
        print(
            f"{Colors.RED}Error: Input path is not a directory: {input_path}{Colors.RESET}"
        )
        sys.exit(1)

    # Find all PDFs
    print(f"{Colors.CYAN}Searching for PDFs in: {input_path}{Colors.RESET}")
    pdf_files = find_pdfs(input_path)

    if not pdf_files:
        print(f"{Colors.YELLOW}No PDF files found in {input_path}{Colors.RESET}")
        sys.exit(0)

    print(f"{Colors.GREEN}Found {len(pdf_files)} PDF file(s) to process{Colors.RESET}")
    for pdf in pdf_files:
        print(f"  - {pdf.relative_to(input_path)}")

    # Initialize model
    print(f"\n{Colors.MAGENTA}Initializing DeepSeek-OCR model...{Colors.RESET}")
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
        )
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    print(f"{Colors.GREEN}Model initialized successfully{Colors.RESET}\n")

    # Process each PDF
    successful = 0
    failed = 0
    processed_output_dirs = []

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}Processing PDF {i}/{len(pdf_files)}{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

        if process_pdf(pdf_path, llm, sampling_params, args.prompt, args.dpi):
            successful += 1
            # Track output directory for second pass
            pdf_basename = pdf_path.stem
            output_dir = pdf_path.parent / f"{pdf_basename}_dpsk"
            processed_output_dirs.append(output_dir)
        else:
            failed += 1

    # Summary of first pass
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}First Pass Complete - PDF Processing{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.GREEN}Successful: {successful}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")

    # Second pass: Process extracted images with Free OCR
    if processed_output_dirs:
        print(f"\n{Colors.MAGENTA}{'='*80}{Colors.RESET}")
        print(
            f"{Colors.MAGENTA}Starting Second Pass - Free OCR on Extracted Images{Colors.RESET}"
        )
        print(f"{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")

        total_images_ocr = 0

        for i, output_dir in enumerate(processed_output_dirs, start=1):
            print(
                f"\n{Colors.CYAN}Processing extracted images from: {output_dir.name}{Colors.RESET}"
            )

            num_processed = process_extracted_images(output_dir, llm, sampling_params)
            total_images_ocr += num_processed

            if num_processed > 0:
                print(
                    f"{Colors.GREEN}✓ Processed {num_processed} image(s){Colors.RESET}"
                )
            else:
                print(f"{Colors.YELLOW}No images to process{Colors.RESET}")

        # Summary of second pass
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}Second Pass Complete - Image OCR{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(
            f"{Colors.GREEN}Total images processed with Free OCR: {total_images_ocr}{Colors.RESET}"
        )

    # Third pass: Translate with LLM
    if processed_output_dirs and not args.skip_translation:
        # Clean up DeepSeek-OCR model to free GPU memory
        print(
            f"\n{Colors.YELLOW}Unloading DeepSeek-OCR model to free GPU memory...{Colors.RESET}"
        )
        del llm
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        print(f"{Colors.GREEN}✓ GPU memory freed{Colors.RESET}\n")
        print(f"\n{Colors.MAGENTA}{'='*80}{Colors.RESET}")
        print(
            f"{Colors.MAGENTA}Starting Third Pass - Swedish Translation{Colors.RESET}"
        )
        print(f"{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")

        total_translations = 0

        for i, output_dir in enumerate(processed_output_dirs, start=1):
            print(
                f"\n{Colors.CYAN}Translating descriptions from: {output_dir.name}{Colors.RESET}"
            )

            num_translated = translate_with_llama3(output_dir, args.llama_model)
            total_translations += num_translated

            if num_translated > 0:
                print(
                    f"{Colors.GREEN}✓ Translated {num_translated} description(s){Colors.RESET}"
                )
            else:
                print(f"{Colors.YELLOW}No descriptions to translate{Colors.RESET}")

        # Summary of third pass
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}Third Pass Complete - Swedish Translation{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(
            f"{Colors.GREEN}Total descriptions translated: {total_translations}{Colors.RESET}"
        )

    # Final summary
    print(f"\n{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.MAGENTA}All Processing Complete{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.GREEN}PDFs successfully processed: {successful}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}PDFs failed: {failed}{Colors.RESET}")
    if processed_output_dirs:
        print(
            f"{Colors.GREEN}Total extracted images OCR'd: {total_images_ocr}{Colors.RESET}"
        )
        if not args.skip_translation:
            print(
                f"{Colors.GREEN}Total descriptions translated to Swedish: {total_translations}{Colors.RESET}"
            )
    print()


if __name__ == "__main__":
    main()
