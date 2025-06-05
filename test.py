import argparse
import os
import numpy as np
import torch
from PIL import Image

from util.preprocess import align_img
from util.load_mats import load_lm3d
from models.facerecon_model import FaceReconModel
from util.landmarks import process_single_landmarks, mask_above_polyline
from util.visualization import save_visualization#


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
    #_, im, lm, _ = align_img(im, lm, lm3d_std)
    transparams, img_new, lm_new, mask_new = align_img(im, lm, lm3d_std)
    #plot_alignment(im, img_new, mask_new)
    im = img_new
    lm = lm_new
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)

    return im, lm

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


if __name__ == '__main__':   
    parser = argparse.ArgumentParser("Test a pre-trained model")
    parser.add_argument("--face_recon_ckpt_path", type=str, default='checkpoints/official/epoch_20.pth')
    parser.add_argument("--parametric_face_model_path", type=str, default='BFM/BFM_model_front.mat')
    parser.add_argument("--img_folder", type=str, default='datasets/examples')
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--sim_lm3d_path", type=str, default='BFM/similarity_Lm3D_all.mat')

    args = parser.parse_args()
    main(0, img_folder=args.img_folder, output_dir=args.output_dir, face_recon_ckpt_path=args.face_recon_ckpt_path, parametric_face_model_path=args.parametric_face_model_path, sim_lm3d_path=args.sim_lm3d_path)
