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
import cv2
from models.facerecon_model import FaceReconModel
from PIL import Image

def calculate_perpendicular_point(p1, p29, p31):
    """
    Calculate a new landmark point on the line between points 1 and 29,
    where the line from this new point to point 31 is perpendicular to the line between points 1 and 29.
    
    Parameters:
        p1 -- numpy.array, coordinates of point 1 [x, y]
        p29 -- numpy.array, coordinates of point 29 [x, y]
        p31 -- numpy.array, coordinates of point 31 [x, y]
    
    Returns:
        new_point -- numpy.array, coordinates of the new point [x, y]
    """
    # Vector from p1 to p29 (direction of the line)
    v_line = p29 - p1
    
    # Normalize the line vector
    v_line_norm = v_line / np.linalg.norm(v_line)
    
    # Vector from p1 to p31
    v_p1_to_p31 = p31 - p1
    
    # Calculate the projection of v_p1_to_p31 onto v_line_norm
    # This gives us the distance along the line from p1 to the perpendicular point
    proj_dist = np.dot(v_p1_to_p31, v_line_norm)
    
    # Calculate the new point by moving proj_dist along the line from p1
    new_point = p1 + proj_dist * v_line_norm
    
    return new_point

def find_line_image_intersection(p1, p2, img_width, img_height):
    """
    Find the intersection of a line defined by two points with the image border.
    
    Parameters:
        p1 -- numpy.array, coordinates of first point [x, y]
        p2 -- numpy.array, coordinates of second point [x, y]
        img_width -- int, width of the image
        img_height -- int, height of the image
    
    Returns:
        intersection_point -- numpy.array, coordinates of the intersection point [x, y]
    """
    # Calculate direction vector of the line
    direction = p2 - p1
    
    # Initialize parameters for the furthest intersection
    max_t = 0
    intersection = None
    
    # Check intersection with left border (x = 0)
    if abs(direction[0]) > 1e-10:  # Avoid division by zero
        t = -p1[0] / direction[0]
        y = p1[1] + t * direction[1]
        if t > 0 and 0 <= y <= img_height - 1 and t > max_t:
            max_t = t
            intersection = np.array([0, y])
    
    # Check intersection with right border (x = img_width - 1)
    if abs(direction[0]) > 1e-10:  # Avoid division by zero
        t = (img_width - 1 - p1[0]) / direction[0]
        y = p1[1] + t * direction[1]
        if t > 0 and 0 <= y <= img_height - 1 and t > max_t:
            max_t = t
            intersection = np.array([img_width - 1, y])
    
    # Check intersection with top border (y = 0)
    if abs(direction[1]) > 1e-10:  # Avoid division by zero
        t = -p1[1] / direction[1]
        x = p1[0] + t * direction[0]
        if t > 0 and 0 <= x <= img_width - 1 and t > max_t:
            max_t = t
            intersection = np.array([x, 0])
    
    # Check intersection with bottom border (y = img_height - 1)
    if abs(direction[1]) > 1e-10:  # Avoid division by zero
        t = (img_height - 1 - p1[1]) / direction[1]
        x = p1[0] + t * direction[0]
        if t > 0 and 0 <= x <= img_width - 1 and t > max_t:
            max_t = t
            intersection = np.array([x, img_height - 1])
    
    # If no intersection found, return the original point
    if intersection is None:
        return p2
    
    return intersection

def find_mask_border_point_at_y(mask, y_coord, from_left=True):
    """
    Find a point on the mask border with the given y-coordinate.
    
    Parameters:
        mask -- numpy.array, the mask image (H, W, C)
        y_coord -- int, the y-coordinate to search at
        from_left -- bool, if True, search from left to right, otherwise from right to left
    
    Returns:
        border_point -- numpy.array, coordinates of the border point [x, y]
                        or None if no border point is found
    """
    H, W = mask.shape[0], mask.shape[1]
    
    # Ensure y_coord is within image bounds
    y = int(np.clip(y_coord, 0, H-1))
    
    # Get the mask row at y_coord
    mask_row = mask[y, :, 0]  # Assuming mask is (H, W, C) and all channels are the same
    
    # Threshold the mask to binary
    binary_mask = (mask_row > 0.5).astype(np.uint8)
    
    # Find the border point
    if from_left:
        # Search from left to right for the first non-zero point
        for x in range(W):
            if binary_mask[x] > 0:
                return np.array([x, y])
    else:
        # Search from right to left for the first non-zero point
        for x in range(W-1, -1, -1):
            if binary_mask[x] > 0:
                return np.array([x, y])
    
    # No border point found
    return None

def draw_numbered_landmarks(img, landmark, color='r', step=2, font_scale=0.3, selected_indices=None):
    """
    Draw landmarks on an image with numbered indices, using the same style as the original draw_landmarks function.
    
    Parameters:
        img -- numpy.array, (H, W, 3), RGB order, range (0, 255)
        landmark -- numpy.array, (68, 2), y direction is opposite to v direction
        color -- str, 'r' or 'b' (red or blue)
        step -- int, size of the landmark dot
        font_scale -- float, scale of the font for the indices
        selected_indices -- list of indices to plot, if None plots all landmarks
    
    Returns:
        img -- numpy.array, (H, W, 3) img with numbered landmarks, RGB order, range (0, 255)
    """
    # Set color based on input parameter, matching the original function
    if color == 'r':
        c = np.array([255., 0, 0])
        cv_color = (0, 0, 255)  # BGR for OpenCV
    else:
        c = np.array([0, 0, 255.])
        cv_color = (255, 0, 0)  # BGR for OpenCV
    
    img = img.copy()
    H, W, _ = img.shape
    
    # Convert landmark coordinates (flip y-axis)
    landmark = landmark.copy()
    landmark[:, 1] = H - 1 - landmark[:, 1]
    landmark = np.round(landmark).astype(np.int32)
    
    # If selected_indices is None, use all indices
    if selected_indices is None:
        indices_to_plot = range(landmark.shape[0])
    else:
        indices_to_plot = selected_indices
    
    # Draw landmarks using the same style as the original function
    for i in indices_to_plot:
        if i >= landmark.shape[0]:
            continue  # Skip if index is out of bounds
            
        x, y = landmark[i]
        
        # Draw the landmark dot using the same approach as the original
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
    
    # Prepare mask visualization (convert to RGB for consistency)
    mask_vis = 255. * pred_mask.permute(0, 2, 3, 1).numpy()
    # If mask is single channel, repeat it to make it RGB
    if mask_vis.shape[-1] == 1:
        mask_vis = np.repeat(mask_vis, 3, axis=-1)
        
    # Add numbered landmarks to the mask visualization
    if pred_lm is not None:
        pred_lm_numpy = pred_lm.numpy()
        # Process each image in the batch
        for i in range(mask_vis.shape[0]):
            # Create a list of indices from 1-16 and 29-35
            selected_indices = list(range(1, 17)) + list(range(29, 36))
            
            # Get landmark points
            p1 = pred_lm_numpy[i, 1]   # Point 1
            p15 = pred_lm_numpy[i, 15]  # Point 15
            
            # Get the mask as a numpy array for border point detection
            mask_numpy = pred_mask.permute(0, 2, 3, 1).numpy()[i]
            
            # Find border points with the same y-coordinates as points 1 and 15
            # For point 1, find the leftmost point on the mask border with the same y-coordinate
            border_point1 = find_mask_border_point_at_y(mask_numpy, p1[1], from_left=True)
            if border_point1 is None:  # Fallback to original point if not found
                border_point1 = p1
            
            # For point 15, find the rightmost point on the mask border with the same y-coordinate
            border_point2 = find_mask_border_point_at_y(mask_numpy, p15[1], from_left=False)
            if border_point2 is None:  # Fallback to original point if not found
                border_point2 = p15
            
            # Create a copy of the landmarks and add the border points at the end
            # Points 70 and 71 are the border points
            landmarks_with_border_points = pred_lm_numpy[i].copy()
            
            # Add the border points if they were found
            if border_point1 is not None:
                landmarks_with_border_points = np.vstack([landmarks_with_border_points, border_point1])
            
            if border_point2 is not None:
                landmarks_with_border_points = np.vstack([landmarks_with_border_points, border_point2])
            
            # Draw the selected landmarks and the border points
            # Include the border points (indices 68 and 69 for border points)
            # Exclude point 16 as requested
            selected_indices = list(range(1, 16)) + list(range(29, 36))
            if border_point1 is not None:
                selected_indices.append(68)  # Index for first border point
            if border_point2 is not None:
                selected_indices.append(69)  # Index for second border point
                
            mask_vis[i] = draw_numbered_landmarks(mask_vis[i], landmarks_with_border_points, 
                                                 color='r', step=2, font_scale=0.3,
                                                 selected_indices=selected_indices)  # 68 and 69 are the indices of the new points
            
            # No line drawing code needed anymore
    
    if gt_lm is not None:
        gt_lm_numpy = gt_lm.numpy()
        pred_lm_numpy = pred_lm.numpy()
        output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
        output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
    
        # Concatenate all four images: input, raw output, output with landmarks, and mask
        output_vis_numpy = np.concatenate((input_img_numpy, 
                            output_vis_numpy_raw, output_vis_numpy, mask_vis), axis=-2)
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
