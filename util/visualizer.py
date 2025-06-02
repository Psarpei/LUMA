import numpy as np
import os
import torch
from . import util

class MyVisualizer:
    def __init__(self, checkpoints_dir):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.checkpoints_dir = checkpoints_dir  # cache the optio
        self.img_dir = os.path.join(self.checkpoints_dir, 'results')
        
    def display_current_results(self, visuals, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
            add_image=True):
        """ Display current results on tensorboard; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        # if (not add_image) and (not save_results): return
        
        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                if add_image:
                    self.writer.add_image(label + '%s_%02d'%(dataset, i + count),
                            image_numpy, total_iters, dataformats='HWC')

                if save_results:
                    save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d'%(epoch, total_iters))
                    print("save_path", save_path)
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    if name is not None:
                        img_path = os.path.join(save_path, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)

    def compute_visuals(self, input_img, pred_face, pred_mask, pred_lm, gt_lm=None):
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

        output_vis = torch.tensor(
                output_vis_numpy / 255., dtype=torch.float32
            ).permute(0, 3, 1, 2)

        return output_vis