import cv2
import numpy as np
from util import util
import torch
import os

def draw_lines_through_landmarks(mask_img, landmark):
    """
    Draw a line from the left border through landmark 68 to 70,
    and from the right border through 69 to 71.
    Args:
        mask_img: numpy array (H, W, 3)
        landmark: numpy array (N, 2), must have at least 72 points
    Returns:
        mask_img with lines drawn
    """
    H, W = mask_img.shape[:2]
    if landmark.shape[0] < 72:
        return mask_img
    pt_68 = landmark[68]
    pt_69 = landmark[69]
    pt_70 = landmark[70]
    pt_71 = landmark[71]
    # Line from left border through 68 to 70
    x0_left = 0
    y0_left = int(round(pt_68[1] + (pt_70[1] - pt_68[1]) * (0 - pt_68[0]) / (pt_70[0] - pt_68[0]) )) if pt_70[0] != pt_68[0] else int(round(pt_68[1]))
    x1_left = int(round(pt_70[0]))
    y1_left = int(round(pt_70[1]))
    cv2.line(mask_img, (x0_left, y0_left), (x1_left, y1_left), (0,255,0), 2)
    # Line from right border through 69 to 71
    x0_right = W-1
    y0_right = int(round(pt_69[1] + (pt_71[1] - pt_69[1]) * ((W-1) - pt_69[0]) / (pt_71[0] - pt_69[0]) )) if pt_71[0] != pt_69[0] else int(round(pt_69[1]))
    x1_right = int(round(pt_71[0]))
    y1_right = int(round(pt_71[1]))
    cv2.line(mask_img, (x0_right, y0_right), (x1_right, y1_right), (0,255,0), 2)
    # Draw polyline from 70 -> 31 -> 32 -> 33 -> 34 -> 35 -> 71
    poly_indices = [70, 31, 32, 33, 34, 35, 71]
    poly_points = np.array([landmark[idx] for idx in poly_indices], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask_img, [poly_points], isClosed=False, color=(0,255,0), thickness=2)
    return mask_img

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
            output_vis_with_landmarks[idx:idx+1] = draw_landmarks(output_vis_with_landmarks[idx:idx+1], gt_lm_numpy[idx:idx+1], 'b', step=2)
            output_vis_with_landmarks[idx:idx+1] = draw_landmarks(output_vis_with_landmarks[idx:idx+1], pred_lm_numpy[idx:idx+1], 'r', step=2)

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