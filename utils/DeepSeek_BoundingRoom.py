# import cv2
# import numpy as np
# import json
# from typing import List, Dict, Tuple, Any
# import matplotlib.pyplot as plt

# def remove_text_from_image(img: np.ndarray) -> np.ndarray:
#     """
#     Remove text from floor plan image using text detection and inpainting.

#     Args:
#         img: Input floor plan image (BGR format).
    
#     Returns:
#         Image with text removed (BGR format).
#     """
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Invert the grayscale image for better thresholding
#     img_inv = cv2.bitwise_not(gray)

#     # Adaptive thresholding to detect text
#     thresh = cv2.adaptiveThreshold(img_inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

#     # Apply morphological closing to remove small gaps in detected text areas
#     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    
#     # Invert the binary image to create the text mask
#     binary = cv2.bitwise_not(closed)

#     # Find contours of potential text regions
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter small contours (likely text) and create a mask
#     text_mask = np.zeros_like(gray)
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         area = cv2.contourArea(cnt)
#         aspect_ratio = w / float(h)
        
#         # Filter based on contour area and aspect ratio (adjust as needed)
#         if area < 1000 and (0.2 < aspect_ratio < 5):
#             cv2.drawContours(text_mask, [cnt], -1, 255, -1)
    
#     # Inpaint text regions using Telea inpainting method
#     inpainted_img = cv2.inpaint(img, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
#     return inpainted_img

# def detect_and_store_rooms(image_path: str, output_img_path: str, output_json_path: str, 
#                          wall_thickness: int = 10, max_door_width: int = 50, 
#                          min_room_area: int = 8000, overlap_threshold: float = 0.7) -> Dict[int, List[Tuple[int, int]]]:
#     """
#     Detect rooms in a floor plan, close door openings, and store room coordinates
    
#     Args:
#         image_path: Path to input floor plan image
#         output_img_path: Path to save visualized output image
#         output_json_path: Path to save room coordinates JSON
#         wall_thickness: Expected wall thickness in pixels
#         max_door_width: Maximum width to consider as a door (pixels)
#         min_room_area: Minimum area to consider as a room (pixels)
#         overlap_threshold: Minimum overlap ratio to consider a room as contained (0.7 = 70%)
    
#     Returns:
#         Dictionary mapping room numbers to their boundary coordinates
#     """
#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Could not load image")
    
#     # Remove text from the image first
#     img_no_text = remove_text_from_image(img)
    
#     # Preprocessing for room detection
#     gray = cv2.cvtColor(img_no_text, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#     kernel = np.ones((wall_thickness, wall_thickness), np.uint8)
#     dilated = cv2.dilate(thresh, kernel, iterations=1)
    
#     # Find contours
#     contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter contours by area and skip the first one (outer boundary)
#     room_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_room_area]
#     if room_contours:
#         room_contours = room_contours[1:]

#     # Remove rooms that are mostly inside another room
#     filtered_contours = []
#     for i, cnt1 in enumerate(room_contours):
#         is_mostly_inside = False
#         for j, cnt2 in enumerate(room_contours):
#             if i == j:
#                 continue
                
#             # Create a blank image to draw contours
#             mask1 = np.zeros_like(gray)
#             mask2 = np.zeros_like(gray)
            
#             # Draw the contours
#             cv2.drawContours(mask1, [cnt1], -1, 255, -1)
#             cv2.drawContours(mask2, [cnt2], -1, 255, -1)
            
#             # Calculate intersection
#             intersection = cv2.bitwise_and(mask1, mask2)
#             intersection_area = cv2.countNonZero(intersection)
#             cnt1_area = cv2.countNonZero(mask1)
            
#             # Check if most of cnt1 is inside cnt2
#             if intersection_area / cnt1_area > overlap_threshold:
#                 is_mostly_inside = True
#                 break
                
#         if not is_mostly_inside:
#             filtered_contours.append(cnt1)

#     room_contours = filtered_contours
    
#     # Prepare output
#     result_img = img.copy()  # Use original image for visualization
#     room_data = {}
    
#     for i, contour in enumerate(room_contours):
#         # Simplify contour
#         epsilon = 0.005 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         approx = approx.reshape(-1, 2)  # Reshape to (N,2) array
        
#         # Find convex hull defects (potential doors)
#         hull = cv2.convexHull(approx, returnPoints=False)
#         defects = cv2.convexityDefects(approx, hull) if hull is not None else None
        
#         # Initialize room boundary with original points
#         room_boundary = approx.tolist()
#         door_segments = []
        
#         if defects is not None:
#             for j in range(defects.shape[0]):
#                 s, e, f, d = defects[j, 0]
#                 start = tuple(approx[s])
#                 end = tuple(approx[e])
                
#                 if d > max_door_width * 256:  # Potential door
#                     # Add door-closing segment to boundary
#                     door_segments.append({"start": start, "end": end})
#                     cv2.line(result_img, start, end, (255, 0, 0), 2)
        
#         # Draw room boundary
#         cv2.drawContours(result_img, [approx.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
        
#         # Label room
#         M = cv2.moments(approx)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             cv2.putText(result_img, f"Room {i+1}", (cX - 20, cY), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
#         # Store room data
#         room_data[i+1] = convert_to_serializable({
#             "boundary": room_boundary,
#             "doors": door_segments,
#             "center": (cX, cY),
#             "area": cv2.contourArea(approx)
#         })
    
#     # Save outputs
#     cv2.imwrite(output_img_path, result_img)
#     with open(output_json_path, 'w') as f:
#         json.dump(convert_to_serializable(room_data), f, indent=2)
    
#     # Visualization
#     plt.figure(figsize=(20, 10))
#     plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
#     plt.subplot(132), plt.imshow(cv2.cvtColor(img_no_text, cv2.COLOR_BGR2RGB)), plt.title('Text Removed')
#     plt.subplot(133), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title('Detected Rooms')
#     plt.show()
    
#     print(f"Detected {len(room_data)} rooms. Results saved to:")
#     print(f"- Image: {output_img_path}")
#     print(f"- Coordinates: {output_json_path}")
    
#     return room_data

# input_image = "../static/uploads/floor_plan8.png"

# room_coordinates = detect_and_store_rooms(
#     image_path=input_image,
#     output_img_path="room_detection_result.png",
#     output_json_path="room_coordinates.json",
#     wall_thickness=15,
#     max_door_width=40
# )

# # To access the coordinates later:
# with open("room_coordinates.json") as f:
#     loaded_data = json.load(f)
#     for room_num, data in loaded_data.items():
#         print(f"\nRoom {room_num}:")
#         print(f"- Boundary points: {len(data['boundary'])} coordinates")
#         print(f"- Doors: {len(data['doors'])}")
#         print(f"- Center point: {data['center']}")


# def plot_room_from_data(room_data, img_width=1000, img_height=1000):
#     """
#     Recreate room visualization from stored data
    
#     Args:
#         room_data: Dictionary containing:
#             - 'boundary': List of [x,y] points
#             - 'doors': List of door segments
#             - 'center': [x,y] coordinates
#         img_width: Width of output image
#         img_height: Height of output image
#     """
#     # Create blank white image
#     img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
#     # Convert boundary points to numpy array
#     boundary = np.array(room_data['boundary'], dtype=np.int32)
    
#     # Draw room boundary (green)
#     cv2.drawContours(img, [boundary], -1, (0, 255, 0), 2)
    
#     # Draw door segments (blue)
#     for door in room_data['doors']:
#         start = tuple(door['start'])
#         end = tuple(door['end'])
#         cv2.line(img, start, end, (255, 0, 0), 2)
    
#     # Draw center point (red)
#     center = tuple(room_data['center'])
#     cv2.circle(img, center, 5, (0, 0, 255), -1)
    
#     # Add room label
#     cv2.putText(img, f"Room Center", (center[0] + 10, center[1]), 
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     # Display
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title('Reconstructed Room')
#     plt.axis('off')
#     plt.show()


import cv2
import numpy as np
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

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

# def remove_text_from_image(img: np.ndarray) -> np.ndarray:
#     """
#     Enhanced text removal from floor plan image using multiple techniques.
    
#     Args:
#         img: Input floor plan image (BGR format).
    
#     Returns:
#         Image with text removed (BGR format).
#     """
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Method 1: Detect text using MSER (Maximally Stable Extremal Regions)
#     mser = cv2.MSER_create()
#     regions, _ = mser.detectRegions(gray)
    
#     # Create mask from MSER regions
#     mser_mask = np.zeros_like(gray)
#     for region in regions:
#         # Filter small regions likely to be text
#         if len(region) < 300:  # Adjust this threshold based on your image
#             x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
#             aspect_ratio = w / float(h)
#             if 0.2 < aspect_ratio < 5:  # Typical text aspect ratios
#                 cv2.drawContours(mser_mask, [region.reshape(-1, 1, 2)], -1, 255, -1)
    
#     # Method 2: Adaptive thresholding for text detection
#     img_inv = cv2.bitwise_not(gray)
#     thresh = cv2.adaptiveThreshold(img_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY, 15, -2)
    
#     # Combine both masks
#     combined_mask = cv2.bitwise_or(mser_mask, thresh)
    
#     # Clean up the mask
#     kernel = np.ones((3, 3), np.uint8)
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     # Inpaint using Navier-Stokes method (better for larger regions)
#     inpainted_img = cv2.inpaint(img, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    
#     return inpainted_img

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
    input_image = "../static/uploads/floor_plan8.png"
    
    room_coordinates = detect_and_store_rooms(
        image_path=input_image,
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