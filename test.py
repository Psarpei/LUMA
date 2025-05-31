"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
import matplotlib.pyplot as plt
from models.facerecon_model import FaceReconModel

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

def main(rank, opt, face_recon_ckpt_path, parametric_face_model_path):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    print(opt)
    print(opt.model)
    print("opt.net_recon", opt.net_recon)
    print("opt.use_last_fc", opt.use_last_fc)
    print("opt.init_path", opt.init_path)

    model = FaceReconModel(opt, face_recon_ckpt_path, parametric_face_model_path)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(opt.img_folder)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            print("%s is not found !!!"%lm_path[i])
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, 0, opt.epoch, dataset=opt.img_folder.split(os.path.sep)[-1], 
            save_results=True, count=i, name=img_name, add_image=False)

        model.save_mesh(os.path.join(visualizer.img_dir, opt.img_folder.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        model.save_coeff(os.path.join(visualizer.img_dir, opt.img_folder.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

if __name__ == '__main__':
    # Get options from the TestOptions class
    opt = TestOptions().parse()  # get test options
    

    face_recon_ckpt_path = 'checkpoints/official/epoch_20.pth'
    parametric_face_model_path = 'BFM/BFM_model_front.mat'
    main(0, opt, face_recon_ckpt_path, parametric_face_model_path)
    
