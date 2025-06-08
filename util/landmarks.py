import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def find_mask_border_point_at_y(mask, y, from_left=True):
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
    _, W = mask.shape[0], mask.shape[1]
   
    # Get the mask row at y_coord
    mask_row = mask[y, :, 0]  # Assuming mask is (H, W, C) and all channels are the same
    
    # Use a lower threshold for better border detection
    binary_mask = (mask_row > 0.1).astype(np.uint8)
    
    if from_left:
        # Start from the left edge of the image and move right
        for x in range(W):
            if binary_mask[x] > 0:
                return np.array([x, y])
    else:
        # Start from the right edge of the image and move left
        for x in range(W-1, -1, -1):
            if binary_mask[x] > 0:
                return np.array([x, y])
    
    # No border point found
    return None

def process_single_landmarks(landmarks, mask):
    """
    Process a single set of landmarks (not a batch).
    
    Parameters:
        landmarks -- numpy.array, shape (N, 2) or (2,)
        mask -- numpy.array, shape (H, W, C)
    
    Returns:
        processed_landmarks -- numpy.array, shape (N+2, 2)
    """
    # Make a copy of the landmarks
    processed_landmarks = landmarks.copy()
    
    # Ensure landmarks is 2D
    if len(processed_landmarks.shape) == 1:
        processed_landmarks = processed_landmarks.reshape(1, 2)
    
    p1 = processed_landmarks[1]   # Point 1
    p15 = processed_landmarks[15]  # Point 15
    
    # Find border points at the y-coordinates of landmarks 1 and 15
    border_point1 = find_mask_border_point_at_y(mask, int(p1[1]), from_left=True)
    border_point2 = find_mask_border_point_at_y(mask, int(p15[1]), from_left=False)

    offset = processed_landmarks[29] - processed_landmarks[33]
    upper_nose_point1 = processed_landmarks[31] + offset
    upper_nose_point2 = processed_landmarks[35] + offset

    # Add the border points to the processed landmarks
    # Ensure they are properly shaped as (1, 2) for vstack
    border_point1 = border_point1.reshape(1, 2)
    border_point2 = border_point2.reshape(1, 2)
    upper_nose_point1 = upper_nose_point1.reshape(1, 2)
    upper_nose_point2 = upper_nose_point2.reshape(1, 2)
    processed_landmarks = np.vstack([processed_landmarks, border_point1, border_point2, upper_nose_point1, upper_nose_point2])

    return processed_landmarks

def mask_above_polyline(mask_img, landmark):
    """
    Set all pixels above the green polyline (full green line) to zero in the mask_img.
    Args:
        mask_img: numpy array (H, W, 3) or (H, W), uint8
        landmark: numpy array (N, 2), must have at least 72 points
    Returns:
        Modified mask_img with pixels above the polyline set to zero
    """
    H, W = mask_img.shape[:2]
    if landmark.shape[0] < 72:
        return mask_img
    pt_30 = landmark[30]
    pt_33 = landmark[33]
    pt_68 = landmark[68]
    pt_69 = landmark[69]

    if pt_30[0] < pt_68[0]:
        # Mask above the line: right border -> 69 -> 33 -> left border
        x_right = W - 1
        y_right = int(round(pt_69[1] + (pt_33[1] - pt_69[1]) * ((W-1) - pt_69[0]) / (pt_33[0] - pt_69[0]) )) if pt_33[0] != pt_69[0] else int(round(pt_69[1]))
        x_left = 0
        y_left = int(round(pt_33[1] + (pt_69[1] - pt_33[1]) * (0 - pt_33[0]) / (pt_69[0] - pt_33[0]) )) if pt_69[0] != pt_33[0] else int(round(pt_33[1]))
        poly_points = np.array([
            [x_right, y_right],
            [int(round(pt_69[0])), int(round(pt_69[1]))],
            [int(round(pt_33[0])), int(round(pt_33[1]))],
            [x_left, y_left],
            [x_left, 0],
            [x_right, 0],
        ], dtype=np.int32)
        poly_points = poly_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask_img, [poly_points], 0)
    elif pt_30[0] > pt_69[0]:
        # Mask above the line: left border -> 68 -> 33 -> right border
        x_left = 0
        y_left = int(round(pt_68[1] + (pt_33[1] - pt_68[1]) * (0 - pt_68[0]) / (pt_33[0] - pt_68[0]) )) if pt_33[0] != pt_68[0] else int(round(pt_68[1]))
        x_right = W - 1
        y_right = int(round(pt_33[1] + (pt_68[1] - pt_33[1]) * ((W-1) - pt_33[0]) / (pt_68[0] - pt_33[0]) )) if pt_68[0] != pt_33[0] else int(round(pt_33[1]))
        poly_points = np.array([
            [x_left, y_left],
            [int(round(pt_68[0])), int(round(pt_68[1]))],
            [int(round(pt_33[0])), int(round(pt_33[1]))],
            [x_right, y_right],
            [x_right, 0],
            [x_left, 0],
        ], dtype=np.int32)
        poly_points = poly_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask_img, [poly_points], 0)
    else:
        # Default/original behavior
        poly_indices = [68, 70, 31, 32, 33, 34, 35, 71, 69]
        poly_points = np.array([landmark[idx] for idx in poly_indices], dtype=np.int32)
        # Add points for left and right border
        poly_points = np.vstack([np.array([0, poly_points[0, 1]]), poly_points, np.array([W-1, poly_points[-1, 1]])])
        # Add points for top border
        poly_points = np.vstack([poly_points, np.array([W-1, 0]), np.array([0, 0])])
        # Close the polygon
        poly_points = np.vstack([poly_points, poly_points[0]])
        poly_points = poly_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask_img, [poly_points], 0)

    return mask_img

def process_mask_with_landmarks(mask, landmarks, dilation_kernel_size=7, dilation_iterations=2, blur_kernel_size=25, blur_iterations=1):
    """
    Post-process a predicted face mask using hole filling, dilation, polyline masking, and smoothing.
    Args:
        mask: numpy array (H, W, 1) or (H, W), uint8 or bool
        landmarks: numpy array (N, 2)
        dilation_kernel_size: int, size of the dilation kernel (default 5)
        dilation_iterations: int, number of dilation iterations (default 2)
        blur_kernel_size: int, size of the Gaussian blur kernel (default 11)
        blur_iterations: int, number of times to apply Gaussian blur (default 3)
    Returns:
        Processed mask as (H, W), uint8, values 0-255, with smooth edges.
    """

    # Ensure mask is (H, W)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]
    mask = mask.astype(np.uint8)
    # 1. Fill holes
    mask_bin = (mask > 0).astype(np.uint8)
    mask_filled = binary_fill_holes(mask_bin).astype(np.uint8)
    # 2. Dilation
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    mask_filled = cv2.dilate(mask_filled, kernel, iterations=dilation_iterations)
    # 3. Polyline masking
    mask_poly = np.expand_dims(mask_filled, axis=2) if mask_filled.ndim == 2 else mask_filled
    mask_poly = mask_above_polyline(mask_poly, landmarks)
    mask_poly = mask_poly.squeeze()
    # 4. Scale to 0-255
    mask_out = (mask_poly * 255).astype(np.uint8)
    # 5. Smoothing
    for _ in range(blur_iterations):
        mask_out = cv2.GaussianBlur(mask_out, (blur_kernel_size, blur_kernel_size), 0)
    return mask_out