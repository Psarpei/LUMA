import numpy as np
import cv2

def calculate_perpendicular_point(point1, point2, reference_point):
    """
    Calculate a point that is the perpendicular projection of a reference point
    onto the line defined by point1 and point2.
    
    Parameters:
        point1, point2 -- numpy.array, coordinates of the points defining the line
        reference_point -- numpy.array, coordinates of the reference point
    
    Returns:
        perpendicular_point -- numpy.array, coordinates of the perpendicular point
    """
    # Vector from point1 to point2
    line_vector = point2 - point1
    
    # Vector from point1 to reference_point
    point_vector = reference_point - point1
    
    # Calculate the projection of point_vector onto line_vector
    line_length_squared = np.sum(line_vector**2)
    
    # Avoid division by zero
    if line_length_squared < 1e-10:
        return point1.copy()
    
    # Calculate the scalar projection
    scalar_projection = np.dot(point_vector, line_vector) / line_length_squared
    
    # Calculate the perpendicular point
    perpendicular_point = point1 + scalar_projection * line_vector
    
    return perpendicular_point

def find_line_image_intersection(point1, point2, width, height):
    """
    Find the intersection of a line with the image border.
    
    Parameters:
        point1, point2 -- numpy.array, coordinates of the points defining the line
        width, height -- int, dimensions of the image
    
    Returns:
        intersection -- numpy.array, coordinates of the intersection point
    """
    # Direction vector of the line
    direction = point2 - point1
    
    # Parameter values for intersections with image borders
    t_values = []
    
    # Check intersection with left border (x = 0)
    if abs(direction[0]) > 1e-10:
        t = -point1[0] / direction[0]
        y_intersect = point1[1] + t * direction[1]
        if 0 <= y_intersect <= height - 1:
            t_values.append((t, np.array([0, y_intersect])))
    
    # Check intersection with right border (x = width - 1)
    if abs(direction[0]) > 1e-10:
        t = (width - 1 - point1[0]) / direction[0]
        y_intersect = point1[1] + t * direction[1]
        if 0 <= y_intersect <= height - 1:
            t_values.append((t, np.array([width - 1, y_intersect])))
    
    # Check intersection with top border (y = 0)
    if abs(direction[1]) > 1e-10:
        t = -point1[1] / direction[1]
        x_intersect = point1[0] + t * direction[0]
        if 0 <= x_intersect <= width - 1:
            t_values.append((t, np.array([x_intersect, 0])))
    
    # Check intersection with bottom border (y = height - 1)
    if abs(direction[1]) > 1e-10:
        t = (height - 1 - point1[1]) / direction[1]
        x_intersect = point1[0] + t * direction[0]
        if 0 <= x_intersect <= width - 1:
            t_values.append((t, np.array([x_intersect, height - 1])))
    
    # Find the intersection with the largest t value (furthest from point1)
    if t_values:
        t_values.sort(key=lambda x: x[0])
        intersection = t_values[-1][1]
    else:
        # No intersection found, return point2
        intersection = point2.copy()
    
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

def process_landmarks(landmarks, mask):
    """
    Process landmarks and add border points at mask edges.
    
    Parameters:
        landmarks -- numpy.array or torch.Tensor, can be:
                    - Single landmark with shape (2,)
                    - Multiple landmarks with shape (N, 2)
                    - Batch of landmarks with shape (B, N, 2)
        mask -- numpy.array or torch.Tensor, mask image
                If tensor: (B, C, H, W) or (C, H, W)
                If numpy: (H, W, C) or (B, H, W, C)
    
    Returns:
        processed_landmarks -- numpy.array, landmarks with border points appended
    """
    import numpy as np
    import torch
    
    # Check if landmarks is a torch tensor and convert to numpy if needed
    is_tensor = False
    if isinstance(landmarks, torch.Tensor):
        is_tensor = True
        landmarks = landmarks.detach().cpu().numpy()
    
    # Check if mask is a torch tensor and convert to numpy if needed
    if isinstance(mask, torch.Tensor):
        # Handle tensor with shape (B, C, H, W) or (C, H, W)
        if len(mask.shape) == 4:  # (B, C, H, W)
            mask_numpy = mask.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:  # (C, H, W)
            mask_numpy = mask.permute(1, 2, 0).detach().cpu().numpy()
    else:
        # Mask is already numpy array with shape (H, W, C) or (B, H, W, C)
        mask_numpy = mask
    
    # Handle different shapes of landmarks
    if len(landmarks.shape) == 3:  # Batch of landmarks (B, N, 2)
        batch_size = landmarks.shape[0]
        processed_landmarks = []
        
        for i in range(batch_size):
            # Process each item in the batch
            if len(mask_numpy.shape) == 4:  # Batch of masks (B, H, W, C)
                mask_i = mask_numpy[i]
            else:  # Single mask (H, W, C)
                mask_i = mask_numpy
            
            # Process landmarks for this item
            landmarks_i = landmarks[i]
            processed_i = process_single_landmarks(landmarks_i, mask_i)
            processed_landmarks.append(processed_i)
        
        # Stack the processed landmarks back into a batch
        processed_landmarks = np.array(processed_landmarks)
    elif len(landmarks.shape) == 2:  # Multiple landmarks (N, 2)
        processed_landmarks = process_single_landmarks(landmarks, mask_numpy)
    else:  # Single landmark (2,)
        # Reshape to (1, 2) and process
        landmarks_reshaped = landmarks.reshape(1, 2)
        processed_landmarks = process_single_landmarks(landmarks_reshaped, mask_numpy)
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        processed_landmarks = torch.from_numpy(processed_landmarks)
    
    return processed_landmarks

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
    
    # Get landmark points for border detection
    # Use safe indexing to avoid out of bounds errors
    if processed_landmarks.shape[0] > 15:
        p1 = processed_landmarks[1]   # Point 1
        p15 = processed_landmarks[15]  # Point 15
    elif processed_landmarks.shape[0] > 1:
        p1 = processed_landmarks[1]   # Point 1
        p15 = processed_landmarks[1]  # Use point 1 as fallback for point 15
    else:
        p1 = processed_landmarks[0]   # Use first point for both
        p15 = processed_landmarks[0]
    
    # Get dimensions of the mask
    H, W = mask.shape[0], mask.shape[1]
    
    # Find border points at the y-coordinates of landmarks 1 and 15
    border_point1 = find_mask_border_point_at_y(mask, p1[1], from_left=True)
    border_point2 = find_mask_border_point_at_y(mask, p15[1], from_left=False)

    if border_point1 is None:  # Fallback to original point if not found
        border_point1 = p1.copy()
    if border_point2 is None:  # Fallback to original point if not found
        border_point2 = p15.copy()

    # Add the border points to the processed landmarks
    # Ensure they are properly shaped as (1, 2) for vstack
    border_point1 = border_point1.reshape(1, 2)
    border_point2 = border_point2.reshape(1, 2)
    processed_landmarks = np.vstack([processed_landmarks, border_point1, border_point2])

    return processed_landmarks

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
    poly_indices = [68, 70, 31, 32, 33, 34, 35, 71, 69]
    poly_points = np.array([landmark[idx] for idx in poly_indices], dtype=np.int32)
    # Add points for left and right border
    poly_points = np.vstack([np.array([0, poly_points[0, 1]]), poly_points, np.array([W-1, poly_points[-1, 1]])])
    # Add points for top border
    poly_points = np.vstack([poly_points, np.array([W-1, 0]), np.array([0, 0])])
    # Close the polygon
    poly_points = np.vstack([poly_points, poly_points[0]])
    poly_points = poly_points.reshape((-1, 1, 2))
    if mask_img.ndim == 3:
        cv2.fillPoly(mask_img, [poly_points], (0, 0, 0))
    else:
        cv2.fillPoly(mask_img, [poly_points], 0)
    return mask_img

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
