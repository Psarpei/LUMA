import os
import numpy as np
import face_alignment
from skimage import io
from glob import glob

def extract_five_landmarks(landmarks_68):
    left_eye = landmarks_68[36:42].mean(axis=0)
    right_eye = landmarks_68[42:48].mean(axis=0)
    nose = landmarks_68[30]
    left_mouth = landmarks_68[48]
    right_mouth = landmarks_68[54]
    return np.stack([left_eye, right_eye, nose, left_mouth, right_mouth])

def save_landmarks_txt(landmarks, txt_path):
    order = [0, 1, 2, 3, 4]  # predefined order
    with open(txt_path, 'w') as f:
        for idx in order:
            x, y = landmarks[idx]
            f.write(f"{x:.3f} {y:.3f}\n")

def process_directory(img_dir):
    output_dir = os.path.join(img_dir, "detections")
    os.makedirs(output_dir, exist_ok=True)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in img_exts:
        image_paths.extend(glob(os.path.join(img_dir, ext)))

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        try:
            input_img = io.imread(img_path)
            preds = fa.get_landmarks(input_img)

            if preds is None:
                print(f"{img_name}: no face detected")
                continue

            landmarks_68 = preds[0]
            landmarks_5 = extract_five_landmarks(landmarks_68)
            save_landmarks_txt(landmarks_5, txt_path)

            print(f"Saved landmarks for {img_name} to {txt_name}")

        except Exception as e:
            print(f"{img_name}: error - {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory of images")
    args = parser.parse_args()
    process_directory(args.directory)
