# Install with CUDA support
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

import argparse
import os
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

#!/usr/bin/env python3
"""
Script to translate PNG image metadata descriptions from English to Swedish
using the gpt-sw3-6.7b-v2-translator model.
"""



def load_translator_model(model_size="Q4_K_M"):
    """Load the Swedish translator model from HuggingFace."""
    model_file = f"gpt-sw3-6-7b-v2-translator-{model_size}.gguf"
    
    model_path = hf_hub_download(
        repo_id="AI-Sweden-Models/gpt-sw3-6.7b-v2-translator-gguf",
        filename=model_file
    )
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=-1,
        verbose=False
    )
    
    return llm


def translate_to_swedish(llm, text):
    """Translate English text to Swedish."""
    prompt = f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:"
    
    response = llm(
        prompt,
        max_tokens=512,
        stop=["<s>", "User:"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()


def process_png_file(file_path, llm, dry_run=False):
    """Process a single PNG file and translate its Description metadata."""
    try:
        img = Image.open(file_path)
        metadata = img.info
        
        if 'Description' not in metadata:
            print(f"  No Description metadata in {file_path}")
            return False
        
        original_desc = metadata['Description']
        print(f"  Original: {original_desc}")
        
        # Translate to Swedish
        translated_desc = translate_to_swedish(llm, original_desc)
        print(f"  Translated: {translated_desc}")
        
        if not dry_run:
            # Save with updated metadata
            png_info = PngInfo()
            for key, value in metadata.items():
                if key == 'Description':
                    png_info.add_text(key, translated_desc)
                else:
                    png_info.add_text(key, value)
            
            img.save(file_path, pnginfo=png_info)
            print(f"  Updated: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False


def process_folder(folder_path, llm, dry_run=False):
    """Recursively process all PNG files in a folder."""
    folder = Path(folder_path)
    png_files = list(folder.rglob("*.png"))
    
    print(f"Found {len(png_files)} PNG files")
    
    processed = 0
    for png_file in png_files:
        print(f"\nProcessing: {png_file}")
        if process_png_file(png_file, llm, dry_run):
            processed += 1
    
    print(f"\nProcessed {processed}/{len(png_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Translate PNG metadata descriptions to Swedish"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder path to process"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="Q4_K_M",
        choices=["Q4", "Q4_K_M", "Q8", "f16"],
        help="Model quantization size (default: Q4_K_M)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview translations without saving"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return
    
    print("Loading translator model...")
    llm = load_translator_model(args.model_size)
    
    print(f"Processing folder: {args.folder}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    process_folder(args.folder, llm, args.dry_run)


if __name__ == "__main__":
    main()