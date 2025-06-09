import os
import argparse
import numpy as np
from glob import glob
from skimage.io import imread
from skimage.transform import resize
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

def load_images_with_paths(img_dir, batch_size):
    img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob(os.path.join(img_dir, ext)))
    img_paths.sort()

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch = []
        for path in batch_paths:
            img = imread(path)
            h, w = img.shape[:2]
            img_resized = resize(img, (256, 256), preserve_range=True, anti_aliasing=True)
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()
            batch.append((img_tensor, w, h, path))
        yield batch

def main():
    parser = argparse.ArgumentParser(description="Batch landmark extraction using face-alignment")
    parser.add_argument("--directory", type=str, required=True, help="Directory with input images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for GPU processing")
    args = parser.parse_args()

    input_dir = args.directory
    output_dir = os.path.join(input_dir, "detections")
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device, face_detector='sfd')

    for batch in tqdm(load_images_with_paths(input_dir, args.batch_size), desc="Processing batches"):
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

        for (w, h), img_path, landmarks_68 in zip(sizes, paths, preds):
            img_name = os.path.basename(img_path)
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_name)

            if landmarks_68 is None or landmarks_68.shape != (68, 2):
                print(f"{img_name}: detected face but landmarks shape {landmarks_68.shape if landmarks_68 is not None else None} is invalid")
                continue

            # Rescale to original size
            scale_x = w / 256.0
            scale_y = h / 256.0
            landmarks_68 = landmarks_68 * np.array([scale_x, scale_y])

            landmarks_5 = extract_five_landmarks(landmarks_68)
            save_landmarks_txt(landmarks_5, txt_path)
            print(f"Saved landmarks for {img_name} to {txt_name}")

if __name__ == "__main__":
    main()
