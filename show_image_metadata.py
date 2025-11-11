import argparse
from pathlib import Path
from PIL import Image

def print_png_metadata(png_path):
    """Print metadata from a PNG file."""
    try:
        with Image.open(png_path) as img:
            print(f"\n{png_path}")
            print("-" * 80)
            if img.info:
                for key, value in img.info.items():
                    print(f"{key}: {value}")
            else:
                print("No metadata found")
    except Exception as e:
        print(f"Error reading {png_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Print PNG metadata from files in a folder")
    parser.add_argument("-i", "--input", required=True, help="Input folder path")
    args = parser.parse_args()

    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        return
    
    png_files = list(input_path.rglob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_path}")
        return
    
    print(f"Found {len(png_files)} PNG file(s)")
    
    for png_file in png_files:
        print_png_metadata(png_file)


if __name__ == "__main__":
    main()