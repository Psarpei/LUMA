"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import argparse
import os
from collections import OrderedDict
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from util.preprocess import align_img
from util.load_mats import load_lm3d
from models.facerecon_model import FaceReconModel
import util.util as util
from util.landmarks import process_landmarks, draw_numbered_landmarks, draw_lines_through_landmarks, mask_above_polyline

# All landmark processing functions have been moved to util/landmarks.py

def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def plot_alignment(img, img_new, mask_new=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_new)
    plt.title("Aligned Image")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    if mask_new is not None:
        plt.imshow(mask_new)
        plt.title("Aligned Mask")
    else:
        plt.title("Aligned Mask (None)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    #_, im, lm, _ = align_img(im, lm, lm3d_std)
    transparams, img_new, lm_new, mask_new = align_img(im, lm, lm3d_std)
    #plot_alignment(im, img_new, mask_new)
    im = img_new
    lm = lm_new
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)

    return im, lm

def save_visualization(input_img, pred_face, pred_mask, pred_lm, processed_landmarks, gt_lm, img_name, output_dir, mask_with_only_lines=None):
    """Compute and save visualization directly without using the visualizer class.
    
    Parameters:
        input_img -- input image tensor
        pred_face -- predicted face tensor
        pred_mask -- predicted mask tensor
        pred_lm -- predicted landmarks tensor
        gt_lm -- ground truth landmarks tensor
        img_name -- name for the output image file
        output_dir -- directory to save the output image    
    """
    # Convert tensors to numpy arrays for visualization
    input_img_numpy = 255. * input_img.permute(0, 2, 3, 1).numpy()
    output_vis = pred_face * pred_mask + (1 - pred_mask) * input_img
    output_vis_numpy_raw = 255. * output_vis.permute(0, 2, 3, 1).numpy()
    
    # Prepare mask visualization (convert to RGB for consistency)
    mask_vis = 255. * pred_mask.permute(0, 2, 3, 1).numpy()
    # If mask is single channel, repeat it to make it RGB
    if mask_vis.shape[-1] == 1:
        mask_vis = np.repeat(mask_vis, 3, axis=-1)
        
    # Add numbered landmarks to the mask visualization using processed landmarks
    if processed_landmarks is not None:
        for i in range(mask_vis.shape[0]):
            lm = processed_landmarks[i]
            # Draw the selected landmarks and the border points
            selected_indices = list(range(1, 16)) + list(range(29, 36))
            if lm.shape[0] > 68:
                selected_indices.append(68)
            if lm.shape[0] > 69:
                selected_indices.append(69)
            # For the last image, also plot landmarks 27 and 28
            if i == mask_vis.shape[0] - 1:
                selected_indices += [27, 28]
                # --- Calculate and plot two new offset points ---
                # Ensure we have enough points
                derived_blue_indices = []
                if lm.shape[0] > 35:
                    offset = lm[29] - lm[33]
                    new_point1 = lm[31] + offset
                    new_point2 = lm[35] + offset
                    # Append new points to lm
                    lm = np.vstack([lm, new_point1, new_point2])
                    derived_blue_indices = [lm.shape[0] - 2, lm.shape[0] - 1]
                # Draw derived blue points (smaller, same style)
                if derived_blue_indices:
                    mask_vis[i] = draw_numbered_landmarks(mask_vis[i], lm, color='b', step=1, font_scale=0.3, selected_indices=derived_blue_indices)
            mask_vis[i] = draw_numbered_landmarks(mask_vis[i], lm, color='r', step=2, font_scale=0.3, selected_indices=selected_indices)

    
    if gt_lm is not None:
        gt_lm_numpy = gt_lm.numpy()
        pred_lm_numpy = pred_lm.numpy()

        # 1. input_img_numpy: input image
        # 2. output_vis_numpy_raw: predicted face only (no landmarks)
        # 3. output_vis_with_landmarks: predicted face + all landmarks (blue and red, step=2)
        # 4. mask_with_derived: mask with only derived blue landmarks (step=2, only last image)

        output_vis_with_landmarks = output_vis_numpy_raw.copy()
        for idx in range(output_vis_numpy_raw.shape[0]):
            # Draw all blue and red points, same size
            output_vis_with_landmarks[idx:idx+1] = util.draw_landmarks(output_vis_with_landmarks[idx:idx+1], gt_lm_numpy[idx:idx+1], 'b', step=2)
            output_vis_with_landmarks[idx:idx+1] = util.draw_landmarks(output_vis_with_landmarks[idx:idx+1], pred_lm_numpy[idx:idx+1], 'r', step=2)

        # Prepare mask visualization for the fourth image:
        # - For the last image: show the original mask with all red points and the two derived blue points (step=2)
        # - For all others: show a blank mask
        mask_with_derived = np.zeros_like(mask_vis)
        for i in range(mask_vis.shape[0]):
            lm = processed_landmarks[i]
            if i == mask_vis.shape[0] - 1:
                # Start with the original mask
                mask_with_derived[i] = mask_vis[i].copy()
                # Draw all red points as before
                selected_indices = list(range(1, 16)) + list(range(29, 36))
                if lm.shape[0] > 68:
                    selected_indices.append(68)
                if lm.shape[0] > 69:
                    selected_indices.append(69)
                mask_with_derived[i] = draw_numbered_landmarks(mask_with_derived[i], lm, color='r', step=2, font_scale=0.3, selected_indices=selected_indices)
                # Calculate and draw derived blue points
                derived_blue_indices = []
                if lm.shape[0] > 35:
                    offset = lm[29] - lm[33]
                    new_point1 = lm[31] + offset
                    new_point2 = lm[35] + offset
                    lm = np.vstack([lm, new_point1, new_point2])
                    derived_blue_indices = [lm.shape[0] - 2, lm.shape[0] - 1]
                if derived_blue_indices:
                    mask_with_derived[i] = draw_numbered_landmarks(mask_with_derived[i], lm, color='b', step=2, font_scale=0.3, selected_indices=derived_blue_indices)
                # Draw lines from left/right border through 68/69 to 70
                mask_with_derived[i] = draw_lines_through_landmarks(mask_with_derived[i], lm)

        # Prepare mask with only lines and polyline (no landmarks) for the 5th image
        mask_with_only_lines = mask_with_only_lines if mask_with_only_lines is not None else np.zeros_like(mask_vis)
        for i in range(mask_vis.shape[0]):
            if i == mask_vis.shape[0] - 1:
                # Use a clean mask from pred_mask (convert to RGB if needed)
                mask_clean = pred_mask[i].detach().cpu().numpy()
                if mask_clean.shape[0] == 1:
                    mask_clean = np.repeat(mask_clean, 3, axis=0)
                mask_clean = (mask_clean * 255).astype(np.uint8)
                mask_clean = np.transpose(mask_clean, (1, 2, 0))
                mask_with_only_lines[i] = mask_clean.copy()
                lm = processed_landmarks[i]
                # Add derived points for correct lines/polyline geometry
                if lm.shape[0] > 35:
                    offset = lm[29] - lm[33]
                    new_point1 = lm[31] + offset
                    new_point2 = lm[35] + offset
                    lm = np.vstack([lm, new_point1, new_point2])
                mask_with_only_lines[i] = mask_above_polyline(mask_with_only_lines[i], lm)
        # All other images remain blank (just zeros or mask background), no landmarks or lines

        # Concatenate all five images: input, raw output, output with landmarks, mask with derived blue points, mask with only lines
        output_vis_numpy = np.concatenate((input_img_numpy, 
                                          output_vis_numpy_raw, 
                                          output_vis_with_landmarks, 
                                          mask_with_derived,
                                          mask_with_only_lines), axis=-2)

    else:
        # Concatenate three images if no landmarks: input, raw output, and mask
        output_vis_numpy = np.concatenate((input_img_numpy, 
                            output_vis_numpy_raw, mask_vis), axis=-2)

    
    # Convert back to tensor format like in the original visualizer
    output_vis = torch.tensor(
        output_vis_numpy / 255., dtype=torch.float32
    ).permute(0, 3, 1, 2)
    
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Process the tensor using the same method as in the original code
    image_numpy = util.tensor2im(output_vis[0])
    
    # Save the image
    img_path = os.path.join(output_dir, f'{img_name}.png')
    util.save_image(image_numpy, img_path)
    
    print(f"Saved visualization to {img_path}")
    return output_vis

def main(rank, img_folder, output_dir, face_recon_ckpt_path, parametric_face_model_path, sim_lm3d_path):
    device = torch.device(rank)
    torch.cuda.set_device(device)

    model = FaceReconModel(face_recon_ckpt_path, parametric_face_model_path, device)
    model.eval()

    im_path, lm_path = get_data_path(img_folder)
    lm3d_std = load_lm3d(sim_lm3d_path) 

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            print("%s is not found !!!"%lm_path[i])
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        
        with torch.no_grad():
            face_shape, pose, gamma_coef, tex_coef = model.proj_img_to_3d(im_tensor.to(device), use_exp=True)
            pred_face, pred_mask, pred_lm = model.proj_3d_to_img(face_shape, pose, gamma_coef,None) #tex_coef)
            print("pred_mask", pred_mask.shape, pred_mask.dtype, pred_mask.min(), pred_mask.max())
            # Process landmarks outside visualization
            pred_lm_numpy = pred_lm.detach().cpu().numpy()
            pred_mask_numpy = pred_mask.permute(0, 2, 3, 1).detach().cpu().numpy()
            processed_landmarks_batch = []
            for j in range(pred_lm_numpy.shape[0]):
                processed_landmarks_batch.append(process_landmarks(pred_lm_numpy[j], pred_mask_numpy[j]))
            processed_landmarks_batch = np.array(processed_landmarks_batch)
            # Flip ground truth landmarks to image coordinates before visualization
            H = im_tensor.shape[2]
            gt_lm_flipped = lm_tensor.clone()
            gt_lm_flipped[..., 1] = H - 1 - gt_lm_flipped[..., 1]
            # --- Calculate mask_with_only_lines for the 5th image outside of save_visualization ---
            from util.landmarks import mask_above_polyline
            mask_vis = 255. * pred_mask.permute(0, 2, 3, 1).numpy()
            B, H, W, _ = mask_vis.shape
            mask_with_only_lines = np.zeros((B, H, W, 3), dtype=np.uint8)
            for k in range(B):
                if k == mask_vis.shape[0] - 1:
                    mask_clean = pred_mask[k].detach().cpu().numpy()
                    if mask_clean.shape[0] == 1:
                        mask_clean = np.repeat(mask_clean, 3, axis=0)
                    mask_clean = (mask_clean * 255).astype(np.uint8)
                    mask_clean = np.transpose(mask_clean, (1, 2, 0))
                    mask_with_only_lines[k] = mask_clean.copy()
                    lm = processed_landmarks_batch[k]
                    # Add derived points for correct lines/polyline geometry
                    if lm.shape[0] > 35:
                        offset = lm[29] - lm[33]
                        new_point1 = lm[31] + offset
                        new_point2 = lm[35] + offset
                        lm = np.vstack([lm, new_point1, new_point2])
                    mask_with_only_lines[k] = mask_above_polyline(mask_with_only_lines[k], lm)
            # Save visualization with both raw and processed landmarks, passing mask_with_only_lines
            save_visualization(im_tensor, pred_face, pred_mask, pred_lm, processed_landmarks_batch, gt_lm_flipped, img_name, output_dir, mask_with_only_lines=mask_with_only_lines)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser("Test a pre-trained model")
    parser.add_argument("--face_recon_ckpt_path", type=str, default='checkpoints/official/epoch_20.pth')
    parser.add_argument("--parametric_face_model_path", type=str, default='BFM/BFM_model_front.mat')
    parser.add_argument("--img_folder", type=str, default='datasets/examples')
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--sim_lm3d_path", type=str, default='BFM/similarity_Lm3D_all.mat')

    args = parser.parse_args()
    main(0, img_folder=args.img_folder, output_dir=args.output_dir, face_recon_ckpt_path=args.face_recon_ckpt_path, parametric_face_model_path=args.parametric_face_model_path, sim_lm3d_path=args.sim_lm3d_path)
