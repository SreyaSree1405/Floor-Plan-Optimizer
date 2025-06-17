import cv2
import pytesseract
import numpy as np
from sklearn.cluster import DBSCAN
import re
import unicodedata
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from typing import Dict, List, Tuple


# ---------- Utility Functions ----------
def show_image(img, title="Image", cmap='gray'):
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = None
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def normalize(text):
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if c.isprintable() or c in '\n\t']).strip()

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

def distance_to_contour(point, contour):
    """Calculate minimum distance from a point to a contour"""
    point = np.array([[point]], dtype=np.float32)
    return cv2.pointPolygonTest(contour, (point[0][0][0], point[0][0][1]), True)

def detect_windows(image_path, max_border_distance=5):
    # Load and clean the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return [], None
    
    display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original Image")
    #cleaned_img = clean_image(img)
    cleaned_img = img
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
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours of potential windows
    contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter windows based on size and distance to border
    window_contours = []
    window_boxes = []
    img_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        area = w * h
        
        # Basic size filtering
        if not (0.2 < aspect_ratio < 5) or not (0 <= area <= 300):
            continue
            
        # Calculate distance to home border
        center = (x + w//2, y + h//2)
        dist = cv2.pointPolygonTest(main_contour, center, True)
        
        # Only keep windows close to the border (inside or outside)
        if abs(dist) <= max_border_distance:
            window_contours.append(cnt)
            window_boxes.append((x, y, x+w, y+h))
    
    # Draw detected windows near border
    result_img = img_rgb.copy()
    for box in window_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw home border again for reference
    cv2.drawContours(result_img, [main_contour], -1, (255, 0, 0), 2)
    display_image(result_img, "Windows Near Home Border")
    
    return window_boxes, result_img, main_contour

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

def remove_text_from_image(img: np.ndarray) -> np.ndarray:
    """
    Conservative text removal that preserves thin structural elements like windows.
    
    Args:
        img: Input floor plan image (BGR format).
    
    Returns:
        Image with text removed (BGR format).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Gentle MSER detection with stricter parameters
    mser = cv2.MSER_create(
        delta=5,                # Smaller value = more stable regions
        min_area=15,            # Minimum text area (smaller than windows)
        max_area=500,           # Maximum text area
        max_variation=0.25      # Lower variation = more uniform regions
    )
    regions, _ = mser.detectRegions(gray)
    
    # Create mask from MSER regions with strict aspect ratio filtering
    mser_mask = np.zeros_like(gray)
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        aspect_ratio = w / float(h)
        
        # Only consider regions that look like text (narrow or square)
        if (0.25 < aspect_ratio < 4) and (15 < w < 200 and 15 < h < 200):
            cv2.drawContours(mser_mask, [region.reshape(-1, 1, 2)], -1, 255, -1)
    
    # Method 2: Gentle adaptive thresholding
    img_inv = cv2.bitwise_not(gray)
    thresh = cv2.adaptiveThreshold(
        img_inv, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11,  # Smaller block size for less aggressive detection
        -2    # Smaller constant
    )
    
    # Combine masks with OR operation
    combined_mask = cv2.bitwise_or(mser_mask, thresh)
    
    # Gentle morphological operations
    kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small isolated pixels (likely noise)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Filter out large components (likely not text)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # Only small areas (text) will be removed
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
    
    # Gentle inpainting
    inpainted_img = cv2.inpaint(img, final_mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
    
    return inpainted_img

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def detect_and_store_rooms(image_path: str, output_img_path: str, output_json_path: str, 
                         wall_thickness: int = 10, max_door_width: int = 50, 
                         min_room_area: int = 15000, overlap_threshold: float = 0.7) -> Dict[int, List[Tuple[int, int]]]:
    """
    Detect rooms in a floor plan, close door openings, and store room coordinates
    
    Args:
        image_path: Path to input floor plan image
        output_img_path: Path to save visualized output image
        output_json_path: Path to save room coordinates JSON
        wall_thickness: Expected wall thickness in pixels
        max_door_width: Maximum width to consider as a door (pixels)
        min_room_area: Minimum area to consider as a room (pixels)
        overlap_threshold: Minimum overlap ratio to consider a room as contained (0.7 = 70%)
    
    Returns:
        Dictionary mapping room numbers to their boundary coordinates
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # Remove text from the image first
    img_no_text = remove_text_from_image(img)
    
    # Preprocessing for room detection
    gray = cv2.cvtColor(img_no_text, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((wall_thickness, wall_thickness), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and skip the first one (outer boundary)
    room_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_room_area]
    if room_contours:
        room_contours = room_contours[1:]

    # Remove rooms that are mostly inside another room
    filtered_contours = []
    for i, cnt1 in enumerate(room_contours):
        is_mostly_inside = False
        for j, cnt2 in enumerate(room_contours):
            if i == j:
                continue
                
            # Create a blank image to draw contours
            mask1 = np.zeros_like(gray)
            mask2 = np.zeros_like(gray)
            
            # Draw the contours
            cv2.drawContours(mask1, [cnt1], -1, 255, -1)
            cv2.drawContours(mask2, [cnt2], -1, 255, -1)
            
            # Calculate intersection
            intersection = cv2.bitwise_and(mask1, mask2)
            intersection_area = cv2.countNonZero(intersection)
            cnt1_area = cv2.countNonZero(mask1)
            
            # Check if most of cnt1 is inside cnt2
            if intersection_area / cnt1_area > overlap_threshold:
                is_mostly_inside = True
                break
                
        if not is_mostly_inside:
            filtered_contours.append(cnt1)

    room_contours = filtered_contours
    
    # Prepare output
    result_img = img.copy()  # Use original image for visualization
    room_data = {}
    
    for i, contour in enumerate(room_contours):
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = approx.reshape(-1, 2)  # Reshape to (N,2) array
        
        # Find convex hull defects (potential doors)
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull) if hull is not None else None
        
        # Initialize room boundary with original points
        room_boundary = approx.tolist()
        door_segments = []
        
        if defects is not None:
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                start = tuple(approx[s])
                end = tuple(approx[e])
                
                if d > max_door_width * 256:  # Potential door
                    # Add door-closing segment to boundary
                    door_segments.append({"start": start, "end": end})
                    cv2.line(result_img, start, end, (255, 0, 0), 2)
        
        # Draw room boundary
        cv2.drawContours(result_img, [approx.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
        
        # Label room
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result_img, f"Room {i+1}", (cX - 20, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Store room data
        room_data[i+1] = convert_to_serializable({
            "boundary": room_boundary,
            "doors": door_segments,
            "center": (cX, cY),
            "area": cv2.contourArea(approx)
        })
    
    # Save outputs
    cv2.imwrite(output_img_path, result_img)
    with open(output_json_path, 'w') as f:
        json.dump(convert_to_serializable(room_data), f, indent=2)
    
    # Visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(132), plt.imshow(cv2.cvtColor(img_no_text, cv2.COLOR_BGR2RGB)), plt.title('Text Removed')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title('Detected Rooms')
    plt.show()
    
    print(f"Detected {len(room_data)} rooms. Results saved to:")
    print(f"- Image: {output_img_path}")
    print(f"- Coordinates: {output_json_path}")
    
    return room_data

# Example usage
if __name__ == "__main__":
    # Optional: Set Tesseract path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Load image
    image_path = '../static/uploads/floor_plan.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit()
    # ---------- OCR + DBSCAN Clustering ----------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(gray, "Grayscale Image")

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    show_image(thresh, "Thresholded Image")

    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

    boxes = []
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append([x, y, x + w, y + h])

    boxes_np = np.array(boxes)
    room_labels_with_coords = {}

    if len(boxes_np) > 0:
        centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in boxes_np])
        avg_size = int((np.mean(boxes_np[:, 2] - boxes_np[:, 0]) + np.mean(boxes_np[:, 3] - boxes_np[:, 1])) / 2)
        eps = max(30, avg_size * 1.2)

        clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
        labels = clustering.labels_

        for label in set(labels):
            cluster_boxes = boxes_np[labels == label]
            x1, y1 = np.min(cluster_boxes[:, 0]), np.min(cluster_boxes[:, 1])
            x2, y2 = np.max(cluster_boxes[:, 2]), np.max(cluster_boxes[:, 3])

            padding = 10
            x1_p, y1_p = max(0, x1 - padding), max(0, y1 - padding)
            x2_p, y2_p = min(gray.shape[1], x2 + padding), min(gray.shape[0], y2 + padding)

            cv2.rectangle(image, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2)

            roi = gray[y1_p:y2_p, x1_p:x2_p]
            roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            roi = cv2.GaussianBlur(roi, (3, 3), 0)

            text = normalize(pytesseract.image_to_string(roi, config='--psm 6').strip())

            if text and not re.fullmatch(r"[\W_]+", text) and (len(text) > 2 or any(char.isdigit() for char in text)):
                if re.search(r"\d+\s?(sq\s?ft|sqft|m2|m²|m\?|sq)", text, re.IGNORECASE):
                    coords = [[x1_p, y1_p], [x2_p, y1_p], [x2_p, y2_p], [x1_p, y2_p]]
                    room_labels_with_coords[text] = coords

    cv2.imwrite('output_with_text_clusters.png', image)
    show_image(image, "Final Image with Text Clusters")

    print("\nCleaned Extracted Room Labels:")
    for label, coords in room_labels_with_coords.items():
        print(f"{label}: Coordinates: {coords}")
    
    room_coordinates = detect_and_store_rooms(
        image_path=image_path,
        output_img_path="room_detection_result.png",
        output_json_path="room_coordinates.json",
        wall_thickness=15,
        max_door_width=40
    )
    
    # To access the coordinates later:
    with open("room_coordinates.json") as f:
        loaded_data = json.load(f)
        for room_num, data in loaded_data.items():
            print(f"\nRoom {room_num}:")
            print(f"- Boundary points: {len(data['boundary'])} coordinates")
            print(f"- Doors: {len(data['doors'])}")
            print(f"- Center point: {data['center']}")

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