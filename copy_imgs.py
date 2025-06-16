import os
import shutil
import argparse
from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def is_image(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS

def copy_images(src_dir, tgt_dir):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)

    if not src_dir.exists() or not src_dir.is_dir():
        raise ValueError(f"Source directory {src_dir} does not exist or is not a directory.")

    tgt_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = Path(root) / file
            if is_image(file_path):
                # Avoid filename collisions
                target_path = tgt_dir / file_path.name
                counter = 1
                while target_path.exists():
                    target_path = tgt_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1

                shutil.copy2(file_path, target_path)
                print(f"Copied: {file_path} -> {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all images from a directory (including subdirectories) to a target directory.")
    parser.add_argument("--source", help="Path to the source directory.")
    parser.add_argument("--target", help="Path to the target directory.")

    args = parser.parse_args()
    copy_images(args.source, args.target)
