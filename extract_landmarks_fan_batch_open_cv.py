import os
import argparse
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import face_alignment

def extract_five_landmarks(landmarks_68):
    left_eye = landmarks_68[36:42].mean(axis=0)
    right_eye = landmarks_68[42:48].mean(axis=0)
    nose = landmarks_68[30]
    left_mouth = landmarks_68[48]
    right_mouth = landmarks_68[54]
    return np.stack([left_eye, right_eye, nose, left_mouth, right_mouth])

def save_landmarks_txt(landmarks, txt_path):
    with open(txt_path, 'w') as f:
        for x, y in landmarks:
            f.write(f"{x:.3f} {y:.3f}\n")

def load_images_with_paths(img_dir, batch_size, resolution):
    img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob(os.path.join(img_dir, ext)))
    img_paths.sort()

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch = []
        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read image: {path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()
            batch.append((img_tensor, w, h, path))
        yield batch

def main():
    parser = argparse.ArgumentParser(description="Batch landmark extraction using face-alignment (OpenCV)")
    parser.add_argument("--directory", type=str, required=True, help="Directory with input images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for GPU processing")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution for image resizing (default: 256)")
    args = parser.parse_args()

    input_dir = args.directory
    output_dir = os.path.join(input_dir, "detections")
    os.makedirs(output_dir, exist_ok=True)

    landmarks_dir = os.path.join(input_dir, "landmarks")
    os.makedirs(landmarks_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device=device, face_detector='sfd')

    for batch in tqdm(load_images_with_paths(input_dir, args.batch_size, args.resolution), desc="Processing batches"):
        if not batch:
            continue
        tensors = torch.stack([item[0] for item in batch]).to(device)
        sizes = [(item[1], item[2]) for item in batch]
        paths = [item[3] for item in batch]

        try:
            preds = fa.get_landmarks_from_batch(tensors)
        except Exception as e:
            print(f"Batch failed: {e}")
            continue

        if preds is None or len(preds) != len(batch):
            print("Batch skipped due to detection failure or mismatched output.")
            continue

        for (w, h), img_path, landmarks_68_3d in zip(sizes, paths, preds):
            img_name = os.path.basename(img_path)
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_name)

            if landmarks_68_3d is None or landmarks_68_3d.shape != (68, 3):
                print(f"{img_name}: detected face but landmarks shape {landmarks_68_3d.shape if landmarks_68_3d is not None else None} is invalid")
                continue

            # Use only x and y for further processing
            landmarks_68 = landmarks_68_3d[:, :2]

            scale_x = w / args.resolution
            scale_y = h / args.resolution
            landmarks_68 = landmarks_68 * np.array([scale_x, scale_y])

            landmarks_5 = extract_five_landmarks(landmarks_68)
            save_landmarks_txt(landmarks_5, txt_path)
            print(f"Saved landmarks for {img_name} to {txt_name}")

            # Invert y-axis for 68 landmarks
            landmarks_68_flipy = landmarks_68.copy()
            landmarks_68_flipy[:, 1] = h - 1 - landmarks_68_flipy[:, 1]

            # Save 68 landmarks in landmarks_dir
            landmarks68_txt_name = os.path.splitext(img_name)[0] + '.txt'
            landmarks68_txt_path = os.path.join(landmarks_dir, landmarks68_txt_name)
            save_landmarks_txt(landmarks_68_flipy, landmarks68_txt_path)
            print(f"Saved 68 landmarks (y flipped) for {img_name} to {landmarks68_txt_name} in 'landmarks' dir")

if __name__ == "__main__":
    main()
