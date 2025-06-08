import os
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from scipy.ndimage import binary_fill_holes

from util.preprocess import align_img, inverse_align_img
from util.load_mats import load_lm3d
from models.facerecon_model import FaceReconModel
from util.landmarks import process_single_landmarks, mask_above_polyline
from util.visualization import save_visualization


def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    transparams, img_new, lm_new, mask_new = align_img(im, lm, lm3d_std)
    orig_im = im.copy()
    if to_tensor:
        im = torch.tensor(np.array(img_new)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm_new).unsqueeze(0)
    else:
        im = img_new
        lm = lm_new
    return im, lm, orig_im, transparams

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
        im_tensor, lm_tensor, orig_im, transparams = read_data(im_path[i], lm_path[i], lm3d_std)
        
        with torch.no_grad():
            face_shape, pose, gamma_coef, tex_coef = model.proj_img_to_3d(im_tensor.to(device), use_exp=True)
            pred_face, pred_mask, pred_lm = model.proj_3d_to_img(face_shape, pose, gamma_coef,None) #tex_coef)
            print("pred_mask", pred_mask.shape, pred_mask.dtype, pred_mask.min(), pred_mask.max())
            # Process landmarks outside visualization
            pred_lm_numpy = pred_lm.detach().cpu().numpy()
            pred_mask_numpy = pred_mask.permute(0, 2, 3, 1).detach().cpu().numpy()
            processed_landmarks_batch = []
            for j in range(pred_lm_numpy.shape[0]):
                print(pred_lm_numpy[j].shape)
                print(pred_mask_numpy.shape)

                processed_landmarks_batch.append(process_single_landmarks(pred_lm_numpy[j], pred_mask_numpy[j]))
            processed_landmarks_batch = np.array(processed_landmarks_batch)

            # --- Calculate mask_with_only_lines for the 5th image outside of save_visualization ---
            mask_vis = 255. * pred_mask.permute(0, 2, 3, 1).numpy()
            B, H, W, _ = mask_vis.shape
            mask_with_only_lines = np.zeros((B, H, W, 1), dtype=np.uint8)
            for k in range(B):
                mask_clean = pred_mask_numpy[k].astype(np.uint8)
                # Fill all holes in the mask using binary_fill_holes (robust to landmark errors)
                mask_bin = (mask_clean[...,0] > 0).astype(np.uint8)
                mask_filled = binary_fill_holes(mask_bin).astype(np.uint8)
                mask_clean[...,0] = mask_filled
                lm = processed_landmarks_batch[k]
                mask_with_only_lines[k] = mask_above_polyline(mask_clean, lm)

            mask_with_only_lines = np.repeat(mask_with_only_lines, 3, axis=3)
            mask_with_only_lines = (mask_with_only_lines * 255).astype(np.uint8)
            # Flip ground truth landmarks to image coordinates before visualization
            H = im_tensor.shape[2]
            gt_lm_flipped = lm_tensor.clone()
            gt_lm_flipped[..., 1] = H - 1 - gt_lm_flipped[..., 1]
            # Save visualization with both raw and processed landmarks, passing mask_with_only_lines
            save_visualization(im_tensor, pred_face, pred_mask, pred_lm, processed_landmarks_batch, gt_lm_flipped, img_name, output_dir, mask_with_only_lines=mask_with_only_lines)

            # --- Transform pred_face back to original image space and save side-by-side plot ---
            # pred_face: (B, 3, 224, 224), orig_im: PIL.Image, transparams: [w0, h0, s, tx, ty]
            pred_face_np = pred_face.detach().cpu().numpy()[0].transpose(1,2,0)  # (224, 224, 3)
            pred_face_np = (pred_face_np * 255).clip(0,255).astype(np.uint8)
            pred_face_orig = inverse_align_img(pred_face_np, transparams)

            # Inverse align the mask_with_only_lines
            mask_lines_np = mask_with_only_lines[0]  # (224, 224, 3), uint8
            mask_lines_orig = inverse_align_img(mask_lines_np, transparams)

            # Create composite image: original * (1-mask) + pred_face * mask
            # mask_lines_orig: (H, W, 3), uint8, values 0 or 255
            mask_bin = (mask_lines_orig > 127).astype(np.uint8)  # (H, W, 3), 0 or 1
            orig_arr = np.array(orig_im).astype(np.uint8)
            # Ensure all images are same size
            if orig_arr.shape != pred_face_orig.shape:
                # Resize pred_face_orig and mask_bin to match orig_arr
                pred_face_orig = cv2.resize(pred_face_orig, (orig_arr.shape[1], orig_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask_bin = cv2.resize(mask_bin, (orig_arr.shape[1], orig_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
            composite = orig_arr * (1 - mask_bin) + pred_face_orig * mask_bin
            composite = composite.astype(np.uint8)

            # Create side-by-side-by-side-by-side image
            side_by_side = np.concatenate([orig_arr, pred_face_orig, mask_lines_orig, composite], axis=1)
            side_by_side_img = Image.fromarray(side_by_side)
            save_path = os.path.join(output_dir, f"{img_name}_orig_predface_masklines_composite.png")
            side_by_side_img.save(save_path)
            print(f"Saved 4-way original, pred_face, mask_with_only_lines, and composite to {save_path}")


if __name__ == '__main__':   
    parser = argparse.ArgumentParser("Test a pre-trained model")
    parser.add_argument("--face_recon_ckpt_path", type=str, default='checkpoints/official/epoch_20.pth')
    parser.add_argument("--parametric_face_model_path", type=str, default='BFM/BFM_model_front.mat')
    parser.add_argument("--img_folder", type=str, default='datasets/examples')
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--sim_lm3d_path", type=str, default='BFM/similarity_Lm3D_all.mat')

    args = parser.parse_args()
    main(0, img_folder=args.img_folder, output_dir=args.output_dir, face_recon_ckpt_path=args.face_recon_ckpt_path, parametric_face_model_path=args.parametric_face_model_path, sim_lm3d_path=args.sim_lm3d_path)
