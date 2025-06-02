"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import argparse
import os
from collections import OrderedDict
# from util.visualizer import MyVisualizer  # No longer needed
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
import matplotlib.pyplot as plt
from util import util
from models.facerecon_model import FaceReconModel
from PIL import Image

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

def save_visualization(input_img, pred_face, pred_mask, pred_lm, gt_lm, img_name, output_dir):
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
    
    if gt_lm is not None:
        gt_lm_numpy = gt_lm.numpy()
        pred_lm_numpy = pred_lm.numpy()
        output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
        output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
    
        output_vis_numpy = np.concatenate((input_img_numpy, 
                            output_vis_numpy_raw, output_vis_numpy), axis=-2)
    else:
        output_vis_numpy = np.concatenate((input_img_numpy, 
                            output_vis_numpy_raw), axis=-2)
    
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
            # Directly save visualization
            save_visualization(im_tensor, pred_face, pred_mask, pred_lm, lm_tensor, img_name, output_dir)

if __name__ == '__main__':   
    parser = argparse.ArgumentParser("Test a pre-trained model")
    parser.add_argument("--face_recon_ckpt_path", type=str, default='checkpoints/official/epoch_20.pth')
    parser.add_argument("--parametric_face_model_path", type=str, default='BFM/BFM_model_front.mat')
    parser.add_argument("--img_folder", type=str, default='datasets/examples')
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--sim_lm3d_path", type=str, default='BFM/similarity_Lm3D_all.mat')

    args = parser.parse_args()
    main(0, img_folder=args.img_folder, output_dir=args.output_dir, face_recon_ckpt_path=args.face_recon_ckpt_path, parametric_face_model_path=args.parametric_face_model_path, sim_lm3d_path=args.sim_lm3d_path)
