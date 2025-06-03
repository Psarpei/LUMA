"""This script contains basic utilities for Deep3DFaceRecon_pytorch
"""
from __future__ import print_function
import numpy as np
import torch
from PIL import Image
try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC
import cv2

        
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1)
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), RESAMPLING_METHOD)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), RESAMPLING_METHOD)
    image_pil.save(image_path)

def draw_landmarks(img, landmark, color='r', step=2):
    """
    Draw landmarks on a batch of images using standard image coordinates (y increases downward).

    Parameters:
        img      -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
        landmark -- numpy.array, (B, 68, 2), standard image coordinates (y increases downward)
        color    -- str, 'r' or 'b' (red or blue)
        step     -- int, size of the landmark dot
    Returns:
        img      -- numpy.array, (B, H, W, 3) img with landmark overlays
    """
    if color == 'r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W, _ = img.shape
    img, landmark = img.copy(), landmark.copy()
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    return img

def draw_numbered_landmarks(img, landmark, color='r', step=2, font_scale=0.3, selected_indices=None):
    """    
    Parameters:
        img -- numpy.array, (H, W, 3), RGB order, range (0, 255)
        landmark -- numpy.array, (68, 2), standard image coordinates (y increases downward)
        color -- str, 'r' or 'b' (red or blue)
        step -- int, size of the landmark dot
        font_scale -- float, scale of the font for the indices
        selected_indices -- list of indices to plot, if None plots all landmarks
    
    Returns:
        img -- numpy.array, image with landmarks drawn
    """
    H, W = img.shape[:2]
    if selected_indices is None:
        selected_indices = list(range(landmark.shape[0]))
    
    if color == 'r':
        c = (255, 0, 0)
        cv_color = (0, 0, 255)
    else:
        c = (0, 0, 255)
        cv_color = (255, 0, 0)
    
    for i in selected_indices:
        x, y = int(round(landmark[i, 0])), int(round(landmark[i, 1]))
        if 0 <= x < W and 0 <= y < H:
            for j in range(-step, step):
                for k in range(-step, step):
                    u = np.clip(x + j, 0, W - 1)
                    v = np.clip(y + k, 0, H - 1)
                    img[v, u] = c
        
        # Add index number next to the landmark
        # Position the text slightly offset from the landmark
        text_x = np.clip(x + step + 1, 0, W - 1)
        text_y = np.clip(y + step + 1, 0, H - 1)
        cv2.putText(img, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, cv_color, 1)
    
    return img
