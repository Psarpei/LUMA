import argparse
import os
from glob import glob

import cv2
from insightface.app import FaceAnalysis

def detect_faces(image_path, face_analyzer):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_analyzer.get(image)

def save_landmarks_txt(landmarks, txt_path):
    # Order: left_eye, right_eye, nose, mouth_left, mouth_right
    order = [0, 1, 2, 3, 4]
    with open(txt_path, 'w') as f:
        for idx in order:
            x, y = landmarks[idx]
            f.write(f"{x:.3f} {y:.3f}\n")

def main():
    parser = argparse.ArgumentParser(description='Detect 5 landmarks using InsightFace buffalo_l model')
    parser.add_argument('--directory', type=str, help='Path to the directory containing images')
    args = parser.parse_args()

    input_dir = args.directory
    output_dir = os.path.join(input_dir, "detections")
    os.makedirs(output_dir, exist_ok=True)

    # Collect image paths
    img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob(os.path.join(input_dir, ext)))

    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    # Initialize the face analysis model
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)  # use -1 for CPU

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        results = detect_faces(img_path, app)
        if results:
            # Use the first face
            landmarks = results[0].kps  # shape (5, 2)
            save_landmarks_txt(landmarks, txt_path)
            print(f"Saved landmarks for {img_name} to {txt_name}")
        else:
            print(f"No face detected in {img_name}")

if __name__ == "__main__":
    main()
