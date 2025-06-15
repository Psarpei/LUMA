import os
import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_landmarks(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    landmarks = np.array([[float(x) for x in line.strip().split()] for line in lines])
    return landmarks


def plot_landmarks(img, landmarks, color, indices=True):
    for i, (x, y) in enumerate(landmarks):
        plt.scatter(x, y, c=color, s=20)
        if indices:
            plt.text(x + 2, y - 2, str(i), color=color, fontsize=7)


def main(img_dir):
    # Get all images in the directory
    img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in img_exts:
        img_paths.extend(sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(ext[1:])]))
    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    # Pick a random image
    img_path = random.choice(img_paths)
    img_name = os.path.basename(img_path)
    print(f"Selected image: {img_name}")

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Load landmarks
    landmarks_dir = os.path.join(img_dir, 'landmarks')
    landmarks_orig_dir = os.path.join(img_dir, 'landmarks_orig')
    lm_name = os.path.splitext(img_name)[0] + '.txt'
    lm_path = os.path.join(landmarks_dir, lm_name)
    lm_orig_path = os.path.join(landmarks_orig_dir, lm_name)

    if not os.path.exists(lm_path):
        print(f"Landmarks file not found: {lm_path}")
        return
    if not os.path.exists(lm_orig_path):
        print(f"Original landmarks file not found: {lm_orig_path}")
        return

    lm = load_landmarks(lm_path)
    lm_orig = load_landmarks(lm_orig_path)

    # Invert y for both sets of landmarks (since both were saved with y inverted)
    lm_plot = lm.copy()
    lm_plot[:, 1] = h - lm_plot[:, 1]
    lm_orig_plot = lm_orig.copy()
    lm_orig_plot[:, 1] = h - lm_orig_plot[:, 1]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plot_landmarks(img, lm_plot, color='blue', indices=True)
    plot_landmarks(img, lm_orig_plot, color='red', indices=True)
    plt.title(f"{img_name}\nBlue: landmarks (inverted y), Red: landmarks_orig")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot 68-point landmarks from 'landmarks' and 'landmarks_orig' on a random image.")
    parser.add_argument("--dir", type=str, help="Directory containing images and landmark subdirectories")
    args = parser.parse_args()
    main(args.dir)
