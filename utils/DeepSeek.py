import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

def display_image(img, title, cmap=None):
    """Helper function to display an image with matplotlib"""
    plt.figure(figsize=(8, 6))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# def clean_image(img):
#     """Remove small black characters and noise from the image"""
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img.copy()
    
#     inverted = cv2.bitwise_not(gray)
#     _, thresh = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     mask = np.zeros_like(gray)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < 100:
#             cv2.drawContours(mask, [cnt], -1, 255, -1)
    
#     kernel = np.ones((3,3), np.uint8)
#     mask = cv2.dilate(mask, kernel, iterations=1)
#     cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
#     return cleaned

def detect_home_border(img):
    """Detect the main outer contour of the home"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours and get the largest one (assuming this is the home)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get the largest contour by area
    main_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask of the home
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    
    # Find the external border (dilate and subtract original)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    border = dilated - mask
    
    return main_contour, border

def detect_windows(image_path, max_border_distance=5, merge_threshold=20):
    # Load and clean the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return [], None
    
    display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original Image")
    cleaned_img = img  # or use clean_image(img) if defined
    display_image(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB), "Cleaned Image")
    
    # Detect home border
    main_contour, border_mask = detect_home_border(cleaned_img)
    if main_contour is None:
        print("Error: Could not detect home border")
        return [], None
    
    # Draw the home border for visualization
    border_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(border_img, [main_contour], -1, (255, 0, 0), 2)
    display_image(border_img, "Detected Home Border")
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours of potential windows
    contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter windows based on size and distance to border
    window_boxes = []
    img_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        
        # Basic size filtering
        if not (0.2 < aspect_ratio < 5) or not (0 <= area <= 300):
            continue
            
        # Calculate distance to home border
        center = (x + w // 2, y + h // 2)
        dist = cv2.pointPolygonTest(main_contour, center, True)
        
        # Only keep windows close to the border (inside or outside)
        if abs(dist) <= max_border_distance:
            window_boxes.append((x, y, x + w, y + h))
    
    # Merge nearby windows
    merged_boxes = []
    used_indices = set()
    
    for i in range(len(window_boxes)):
        if i in used_indices:
            continue
        x1, y1, x2, y2 = window_boxes[i]
        current_box = [x1, y1, x2, y2]
        
        # Check for nearby windows
        for j in range(i + 1, len(window_boxes)):
            if j in used_indices:
                continue
            nx1, ny1, nx2, ny2 = window_boxes[j]
            
            # Calculate distance between current box and nearby box
            dx = max(x1 - nx2, nx1 - x2, 0)
            dy = max(y1 - ny2, ny1 - y2, 0)
            distance = max(dx, dy)
            
            if distance < merge_threshold:
                # Merge the boxes (expand current_box to include the nearby box)
                current_box[0] = min(current_box[0], nx1)
                current_box[1] = min(current_box[1], ny1)
                current_box[2] = max(current_box[2], nx2)
                current_box[3] = max(current_box[3], ny2)
                used_indices.add(j)
        
        merged_boxes.append(current_box)
        used_indices.add(i)
    
    # Draw detected windows (merged or not)
    result_img = img_rgb.copy()
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw home border again for reference
    cv2.drawContours(result_img, [main_contour], -1, (255, 0, 0), 2)
    display_image(result_img, "Windows Near Home Border (Merged if Close)")
    
    return merged_boxes, result_img, main_contour

def distance_to_contour(point, contour):
    """Calculate minimum distance from a point to a contour"""
    point = np.array([[point]], dtype=np.float32)
    return cv2.pointPolygonTest(contour, (point[0][0][0], point[0][0][1]), True)

# def detect_windows(image_path, max_border_distance=5):
#     # Load and clean the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Could not load image")
#         return [], None
    
#     display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original Image")
#     #cleaned_img = clean_image(img)
#     cleaned_img = img
#     display_image(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB), "Cleaned Image")
    
#     # Detect home border
#     main_contour, border_mask = detect_home_border(cleaned_img)
#     if main_contour is None:
#         print("Error: Could not detect home border")
#         return [], None
    
#     # Draw the home border for visualization
#     border_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB).copy()
#     cv2.drawContours(border_img, [main_contour], -1, (255, 0, 0), 2)
#     display_image(border_img, "Detected Home Border")
    
#     # Convert to grayscale and threshold
#     gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
#     # Morphological operations
#     kernel = np.ones((3,3), np.uint8)
#     opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     # Find contours of potential windows
#     contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter windows based on size and distance to border
#     window_contours = []
#     window_boxes = []
#     img_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
    
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = float(w)/h
#         area = w * h
        
#         # Basic size filtering
#         if not (0.2 < aspect_ratio < 5) or not (0 <= area <= 300):
#             continue
            
#         # Calculate distance to home border
#         center = (x + w//2, y + h//2)
#         dist = cv2.pointPolygonTest(main_contour, center, True)
        
#         # Only keep windows close to the border (inside or outside)
#         if abs(dist) <= max_border_distance:
#             window_contours.append(cnt)
#             window_boxes.append((x, y, x+w, y+h))
    
#     # Draw detected windows near border
#     result_img = img_rgb.copy()
#     for box in window_boxes:
#         x1, y1, x2, y2 = box
#         cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     # Draw home border again for reference
#     cv2.drawContours(result_img, [main_contour], -1, (255, 0, 0), 2)
#     display_image(result_img, "Windows Near Home Border")
    
#     return window_boxes, result_img, main_contour

def pair_windows_by_alignment(window_boxes, main_contour, max_diff=4, min_distance=20, max_distance=200):
    """
    Pair windows with alignment constraints and proximity to home border
    """
    horizontal_groups = defaultdict(list)
    vertical_groups = defaultdict(list)
    
    for i, (x1, y1, x2, y2) in enumerate(window_boxes):
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Group by y alignment (horizontal)
        matched_y = None
        for y in horizontal_groups:
            if abs(y - center_y) <= max_diff:
                matched_y = y
                break
        if matched_y is not None:
            horizontal_groups[matched_y].append((i, center_x))
        else:
            horizontal_groups[center_y].append((i, center_x))
        
        # Group by x alignment (vertical)
        matched_x = None
        for x in vertical_groups:
            if abs(x - center_x) <= max_diff:
                matched_x = x
                break
        if matched_x is not None:
            vertical_groups[matched_x].append((i, center_y))
        else:
            vertical_groups[center_x].append((i, center_y))
    
    # Filter groups
    horizontal_groups = {k: sorted(v, key=lambda x: x[1]) 
                        for k, v in horizontal_groups.items() if len(v) >= 2}
    vertical_groups = {k: sorted(v, key=lambda x: x[1]) 
                      for k, v in vertical_groups.items() if len(v) >= 2}
    
    all_pairs = []
    used_windows = set()
    
    # Process horizontal groups
    for y, group in horizontal_groups.items():
        for i in range(len(group) - 1):
            idx1, x1 = group[i]
            idx2, x2 = group[i + 1]
            
            if idx1 in used_windows or idx2 in used_windows:
                continue
                
            distance = abs(x2 - x1)
            if min_distance <= distance <= max_distance:
                all_pairs.append((idx1, idx2, "horizontal", distance))
                used_windows.add(idx1)
                used_windows.add(idx2)
    
    # Process vertical groups
    for x, group in vertical_groups.items():
        for i in range(len(group) - 1):
            idx1, y1 = group[i]
            idx2, y2 = group[i + 1]
            
            if idx1 in used_windows or idx2 in used_windows:
                continue
                
            distance = abs(y2 - y1)
            if min_distance <= distance <= max_distance:
                all_pairs.append((idx1, idx2, "vertical", distance))
                used_windows.add(idx1)
                used_windows.add(idx2)
    
    return all_pairs

def draw_paired_windows(img, window_boxes, paired_windows, main_contour):
    """Draw paired windows with home border"""
    result_img = img.copy()
    
    # Draw home border
    cv2.drawContours(result_img, [main_contour], -1, (255, 0, 0), 2)
    
    # Draw all windows
    for i, (x1, y1, x2, y2) in enumerate(window_boxes):
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, str(i), (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw pairs
    for i, j, alignment, distance in paired_windows:
        box1 = window_boxes[i]
        box2 = window_boxes[j]
        center1 = ((box1[0] + box1[2])//2, (box1[1] + box1[3])//2)
        center2 = ((box2[0] + box2[2])//2, (box2[1] + box2[3])//2)
        
        color = (255, 0, 0) if alignment == "horizontal" else (255, 140, 0)
        cv2.line(result_img, center1, center2, color, 2)
        
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2
        cv2.putText(result_img, f"{distance:.1f}px", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result_img, f"{distance:.1f}px", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result_img

# Example usage
image_path = '../static/uploads/floor_plan8.png'
window_boxes, result_image, main_contour = detect_windows(image_path, max_border_distance=20)

print("Detected Window Coordinates (x1, y1, x2, y2):")
for i, coords in enumerate(window_boxes):
    print(f"Window {i}: {coords}")

# Pair windows
paired_windows = pair_windows_by_alignment(
    window_boxes, 
    main_contour,
    max_diff=4, 
    min_distance=20, 
    max_distance=200
)

print("\nPaired Windows:")
for i, j, alignment, distance in paired_windows:
    print(f"Pair: Window {i} and Window {j} ({alignment} alignment, {distance:.1f}px)")

# Draw final result
paired_image = draw_paired_windows(result_image, window_boxes, paired_windows, main_contour)
display_image(paired_image, "Final Result: Windows Paired Near Home Border")