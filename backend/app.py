from flask import Flask, request, jsonify, send_file, render_template
import cv2
import pytesseract
import numpy as np
from sklearn.cluster import DBSCAN
import re
import unicodedata
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
from collections import defaultdict
import json
from typing import Dict, List, Tuple
import os
import io
import base64
from werkzeug.utils import secure_filename
from flask_cors import CORS 
import base64
import matplotlib.patches as patches
import random
import math
import pandas as pd
from math import pi
from flask import send_from_directory

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'uploads')
os.makedirs(upload_folder, exist_ok=True)

app = Flask(__name__)
CORS(app) 

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ---------- Utility Functions ----------
def show_image(img, title="Image", cmap='gray'):
    """Save matplotlib figure to file and return filename"""
    filename = f"{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = None
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filepath)
    plt.close()
    return filename

def normalize(text):
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if c.isprintable() or c in '\n\t']).strip()

def display_image(img, title, cmap=None):
    """Save matplotlib figure to file and return filename"""
    filename = f"{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    plt.figure(figsize=(8, 6))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filename

def image_to_base64(img):
    """Convert an OpenCV image to base64 encoded string"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def save_and_encode_image(img, filename):
    """Save image to file and return base64 encoding"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    cv2.imwrite(filepath, img)
    return {
        'filepath': f'/static/outputs/{filename}',
        'base64': image_to_base64(img)
    }

def perform_ocr_clustering(image_path, tesseract_path=None, debug=False, draw_boxes=True):
    """
    Extract room information from a floor plan image using OCR.
    
    Args:
        image_path (str): Path to the floor plan image
        tesseract_path (str, optional): Path to Tesseract executable
        debug (bool, optional): Whether to display debug information and save output image
        draw_boxes (bool, optional): Whether to draw bounding boxes on the output image
    
    Returns:
        dict: Dictionary containing room data organized by room type and ID with detailed
              bounding box coordinates for each identified room label
    """
    # Set Tesseract path if provided
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Load and check image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image from {image_path}")
    
    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # OCR configuration
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Extract bounding boxes for non-empty text entries
    boxes = []
    for i in range(len(data['text'])):
        if data['text'][i].strip() != "":
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            boxes.append([x, y, x + w, y + h])  # (x1, y1, x2, y2)
    
    boxes_np = np.array(boxes)
    extracted_texts = []
    
    # Cluster boxes using DBSCAN
    if len(boxes_np) > 0:
        centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in boxes_np])
        widths = boxes_np[:, 2] - boxes_np[:, 0]
        heights = boxes_np[:, 3] - boxes_np[:, 1]
        avg_size = int((np.mean(widths) + np.mean(heights)) / 2)
        eps = max(30, avg_size * 1.2)
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
        labels = clustering.labels_
        
        for label in set(labels):
            cluster_boxes = boxes_np[labels == label]
            x1 = np.min(cluster_boxes[:, 0])
            y1 = np.min(cluster_boxes[:, 1])
            x2 = np.max(cluster_boxes[:, 2])
            y2 = np.max(cluster_boxes[:, 3])
            
            # Expand box slightly for better OCR
            padding = 10
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(gray.shape[1], x2 + padding)
            y2_p = min(gray.shape[0], y2 + padding)
            
            # Crop and preprocess the expanded box
            roi = gray[y1_p:y2_p, x1_p:x2_p]
            roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            
            # OCR on the cropped region
            text = pytesseract.image_to_string(roi, config='--psm 6').strip()
            if text:
                extracted_texts.append((text, (x1_p, y1_p, x2_p, y2_p)))  # Store text with coordinates
                # Draw rectangle and label on debug image
                if debug and draw_boxes:
                    # Draw bounding box
                    cv2.rectangle(image, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2)
                    
                    # Draw text label
                    cv2.putText(image, text, (x1_p, y1_p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                    
                    # Draw center point
                    center_x = (x1_p + x2_p) // 2
                    center_y = (y1_p + y2_p) // 2
                    cv2.circle(image, (center_x, center_y), 3, (255, 0, 0), -1)
                    
                    # Label coordinates
                    coord_text = f"({x1_p},{y1_p})"
                    cv2.putText(image, coord_text, (x1_p, y2_p + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Process extracted text
    if debug:
        print("Extracted text before cleaning:")
        for text, _ in extracted_texts:
            print(f"  - {text}")
    
    # Extract room data from text
    room_data = _extract_room_data(extracted_texts)
    
    # Debug visualization
    if debug:
        # Save and show result
        output_path = os.path.splitext(image_path)[0] + "_output.png"
        cv2.imwrite(output_path, image)
        
        # Display the image using Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Floor Plan OCR Results")
        plt.show()
        
        # Print room information
        print("\nExtracted Room Data:")
        if 'by_id' in room_data and room_data['by_id']:
            for room_id, room_info in room_data['by_id'].items():
                bbox = room_info['bounding_box']
                print(f"{room_id}: Area: {room_info['area']} {room_info['unit']}")
                print(f"  Bounding Box: x1={bbox['x1']}, y1={bbox['y1']}, x2={bbox['x2']}, y2={bbox['y2']}")
                print(f"  Dimensions: {bbox['width']}x{bbox['height']} px, Center: ({bbox['center_x']}, {bbox['center_y']})")
        else:
            print("No room data extracted.")
    
    # Store original image in the return object for reference
    if debug:
        room_data['debug_image'] = image.copy()
    
    return room_data

def _clean_and_normalize_text(text):
    """
    Normalize and clean OCR-extracted text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if c.isprintable()).strip()
    return re.sub(r'\s+', ' ', text)

def _extract_room_data(text_items):
    """
    Extract structured room information with proper area parsing.
    Handles multiple instances of the same room type (e.g., multiple bedrooms).
    
    Args:
        text_items: List of (text, coordinates) tuples
        
    Returns:
        Dictionary with room data organized by room type and ID
    """
    room_pattern = re.compile(
        r'(?P<name>[A-Za-z]+)\s*'       # Room name
        r'(?P<number>\d{1,2}\s+)?'      # Optional room number (1-2 digits followed by space)
        r'(?P<area>\d{2,})\s*'          # Area value (2+ digits)
        r'(?P<unit>s?q?\.?\s?f?t?|m²?)', # Unit
        re.IGNORECASE
    )
    
    rooms = {}  # Dictionary to store rooms by ID
    room_types = defaultdict(list)  # Dictionary to group rooms by type
    junk_pattern = re.compile(r'^[\W_]+$')
    
    for text, coords in text_items:
        text = _clean_and_normalize_text(text)
        
        # Skip junk text
        if len(text) < 3 or junk_pattern.match(text):
            continue
            
        # Try to extract room information
        match = room_pattern.search(text)
        if match:
            name = match.group('name').title()
            number = match.group('number')
            area = match.group('area')
            unit = match.group('unit').replace('?', '²').lower()
            
            # Only consider it a room number if it's 1-2 digits followed by space
            # Otherwise treat it as part of the area
            if number and len(number.strip()) <= 2:
                room_number = int(number.strip())
                room_id = f"{name} {room_number}"
            else:
                # If no explicit number, check if we already have rooms of this type
                existing_count = len(room_types[name])
                room_number = existing_count + 1
                room_id = f"{name} {room_number}" if existing_count > 0 else name
                
                if number:  # If we captured something that's not a valid room number
                    area = number.strip() + area  # Combine with area
                
            area_value = int(area)
            
            # Standardize units
            if 'sq' in unit or 'sf' in unit:
                unit = 'sq ft'
            elif 'm' in unit:
                unit = 'm²'
            
            # Create room data structure
            room_data = {
                'area': area_value,
                'unit': unit,
                'coordinates': [
                    [int(coords[0]), int(coords[1])],
                    [int(coords[2]), int(coords[1])],
                    [int(coords[2]), int(coords[3])],
                    [int(coords[0]), int(coords[3])]
                ],
                'bounding_box': {
                    'x1': int(coords[0]),  # Left
                    'y1': int(coords[1]),  # Top
                    'x2': int(coords[2]),  # Right
                    'y2': int(coords[3]),  # Bottom
                    'width': int(coords[2]) - int(coords[0]),
                    'height': int(coords[3]) - int(coords[1]),
                    'center_x': (int(coords[0]) + int(coords[2])) // 2,
                    'center_y': (int(coords[1]) + int(coords[3])) // 2
                },
                'raw_text': text,
                'type': name  # Adding type field for compatibility
            }
            
            # Store by room ID
            rooms[room_id] = room_data
            
            # Add to the room type grouping
            room_types[name].append(room_data)
    
    # Add a list of all rooms for compatibility
    room_list = []
    for room_id, room_data in rooms.items():
        room_data_copy = room_data.copy()
        room_data_copy['id'] = room_id
        room_list.append(room_data_copy)
    
    # Return structured room data
    return {
        'by_id': rooms,           # Access by room ID: "Bedroom 1", "Kitchen", etc.
        'by_type': room_types,    # Access by room type: ["Bedroom", "Kitchen", etc.]
        'rooms': room_list        # Flat list of all rooms
    }

def detect_home_border(img):
    """Detect the main outer contour of the home"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours and get the largest one (assuming this is the home)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
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

def merge_all_paired_windows(window_data):
    windows = window_data["windows"]
    paired_windows = window_data["paired_windows"]
    merged_windows = []

    for pair in paired_windows:
        idx1, idx2, orientation, _ = pair
        window1 = windows[idx1]
        window2 = windows[idx2]

        if orientation == "vertical":
            # Vertical merge: stack windows vertically (min y1, max y2)
            merged_window = [
                min(window1[0], window2[0]),  # left (x1)
                min(window1[1], window2[1]),  # top (y1)
                max(window1[2], window2[2]),  # right (x2)
                max(window1[3], window2[3])   # bottom (y2)
            ]
        else:  # horizontal
            # Horizontal merge: place windows side by side (min x1, max x2)
            merged_window = [
                min(window1[0], window2[0]),  # left (x1)
                min(window1[1], window2[1]),  # top (y1)
                max(window1[2], window2[2]),  # right (x2)
                max(window1[3], window2[3])   # bottom (y2)
            ]

        merged_windows.append(merged_window)

    return merged_windows

def detect_windows(image_path, max_border_distance=5):
    # Load and clean the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return [], None, None
    
    img_filename = display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original Image")
    cleaned_img = img
    cleaned_filename = display_image(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB), "Cleaned Image")
    
    # Detect home border
    main_contour, border_mask = detect_home_border(cleaned_img)
    if main_contour is None:
        print("Error: Could not detect home border")
        return [], None, None
    
    # Draw the home border for visualization
    border_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(border_img, [main_contour], -1, (255, 0, 0), 2)
    border_filename = display_image(border_img, "Detected Home Border")
    
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
    result_filename = display_image(result_img, "Windows Near Home Border")
    
    result_images = {
        "original": img_filename,
        "cleaned": cleaned_filename,
        "border": border_filename,
        "windows": result_filename
    }
    
    return window_boxes, result_img, main_contour, result_images

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
        Dictionary mapping room numbers to their boundary coordinates, and a dictionary of image paths
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
    
    # Create visualization for the web interface
    plt.figure(figsize=(20, 10))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(132), plt.imshow(cv2.cvtColor(img_no_text, cv2.COLOR_BGR2RGB)), plt.title('Text Removed')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title('Detected Rooms')
    comparison_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    
    image_files = {
        'original': 'original_image.png',
        'text_removed': 'text_removed.png',
        'detected_rooms': os.path.basename(output_img_path),
        'comparison': 'room_comparison.png'
    }
    
    # Save individual images for the API
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'original_image.png'), img)
    cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'text_removed.png'), img_no_text)
    
    return room_data, image_files

def draw_all_coordinates(original_image_path, result_data, output_filename="combined_analysis.png"):
    """
    Draw all detected coordinates (rooms, windows, texts) on the original floor plan image
    
    Args:
        original_image_path (str): Path to the original floor plan image
        result_data (dict): Dictionary containing detection results with coordinates
        output_filename (str): Name of the output file
    
    Returns:
        str: Path to the saved output image
    """
    import cv2
    import numpy as np
    import os
    
    # Load original image
    image = cv2.imread(original_image_path)
    if image is None:
        raise ValueError(f"Could not load image from {original_image_path}")
    
    # Create a copy for drawing
    result_image = image.copy()
    
    # Color definitions (BGR format)
    colors = {
        'room': (0, 255, 0),      # Green
        'window': (255, 0, 0),    # Blue
        'paired_window': (0, 0, 255),  # Red
        'text': (255, 165, 0),    # Orange
        'door': (128, 0, 128)     # Purple
    }
    
    # Draw rooms
    if 'room_detection' in result_data:
        # Try different possible structures for room data
        room_data = result_data['room_detection']
        
        # Option 1: Check if nested under 'data'
        if 'data' in room_data and 'rooms' in room_data['data']:
            rooms = room_data['data']['rooms']
        # Option 2: Check if rooms are directly under room_detection
        elif 'rooms' in room_data:
            rooms = room_data['rooms']
        # Option 3: Check if room_detection itself is a list of rooms
        elif isinstance(room_data, list):
            rooms = room_data
        else:
            rooms = []
        
        for room in rooms:
            if 'coordinates' in room:
                points = np.array(room['coordinates'], dtype=np.int32)
                cv2.polylines(result_image, [points], True, colors['room'], 2)
                
                # Add room label
                if 'type' in room:
                    text_pos = points.mean(axis=0).astype(int)
                    cv2.putText(result_image, room['type'], tuple(text_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['room'], 2)
    
    # Draw windows
    if 'window_detection' in result_data:
        window_data = result_data['window_detection']
        
        # Draw individual windows
        if 'windows' in window_data:
            windows = window_data['windows']
            for window in windows:
                x1, y1, x2, y2 = [int(coord) for coord in window]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), colors['window'], 2)
        
        # Draw paired windows with connecting lines
        if 'paired_windows' in window_data and 'windows' in window_data:
            paired_windows = window_data['paired_windows']
            windows = window_data['windows']
            
            for pair in paired_windows:
                i, j, alignment, _ = pair
                
                if i < len(windows) and j < len(windows):
                    window_i = windows[i]
                    window_j = windows[j]
                    
                    # Calculate centers
                    center_i = (
                        int((window_i[0] + window_i[2]) / 2),
                        int((window_i[1] + window_i[3]) / 2)
                    )
                    center_j = (
                        int((window_j[0] + window_j[2]) / 2),
                        int((window_j[1] + window_j[3]) / 2)
                    )
                    
                    # Draw connecting line
                    cv2.line(result_image, center_i, center_j, colors['paired_window'], 2)
    
    # Draw text labels
    if 'text_extraction' in result_data:
        text_data = result_data['text_extraction']
        
        # Option 1: Check if under 'labels'
        if 'labels' in text_data:
            text_labels = text_data['labels']
        # Option 2: Check if it's a direct list
        elif isinstance(text_data, list):
            text_labels = text_data
        else:
            text_labels = []
            
        for label in text_labels:
            if 'bbox' in label:
                x, y, w, h = label['bbox']
                cv2.rectangle(result_image, (x, y), (x + w, y + h), colors['text'], 2)
                cv2.putText(result_image, label.get('text', ''), (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
    
    # Ensure output directory exists
    output_dir = os.path.join(".", "static", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the result image
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, result_image)
    
    # Return the web-accessible path
    return f"/static/outputs/{output_filename}"

def is_point_in_polygon(point, polygon):
    x, y = point
    inside = False
    
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
        j = i
    
    return inside

def get_text_center(label):
    return [
        label["bounding_box"]["center_x"],
        label["bounding_box"]["center_y"]
    ]

def calculate_overlap_area(label, room):
    bbox = label["bounding_box"]
    corners = [
        [bbox["x1"], bbox["y1"]],
        [bbox["x2"], bbox["y1"]],
        [bbox["x2"], bbox["y2"]],
        [bbox["x1"], bbox["y2"]]
    ]
    
    corners_inside = [corner for corner in corners if is_point_in_polygon(corner, room["boundary"])]
    return len(corners_inside) / 4

def distance_to_line_segment(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    
    if len_sq != 0:
        param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = x - xx
    dy = y - yy
    
    return math.sqrt(dx * dx + dy * dy)

def is_window_on_room_boundary(window, room, distance_threshold=20):
    window_center = [(window[0] + window[2]) / 2, (window[1] + window[3]) / 2]
    
    boundary = room["boundary"]
    n = len(boundary)
    for i in range(n):
        j = (i + 1) % n
        start = boundary[i]
        end = boundary[j]
        
        distance = distance_to_line_segment(window_center, start, end)
        if distance < distance_threshold:
            return True
    
    return False

def analyze_rooms_and_windows(room_detection, window_detection, text_extraction):
    result = {}
    
    # Step 1: Associate text labels with rooms
    rooms_with_labels = {}
    
    for room_id, room in room_detection["data"].items():
        rooms_with_labels[room_id] = {
            **room,
            "name": None,
            "label_overlap": 0
        }
        
        # Find the best matching label for this room
        for label_id, label in text_extraction["labels"]["by_id"].items():
            # Skip labels that aren't room types
            if "type" not in label:
                continue
            
            overlap_amount = calculate_overlap_area(label, room)
            
            # If this label has more overlap than any previous label, associate it with this room
            if overlap_amount > rooms_with_labels[room_id]["label_overlap"]:
                rooms_with_labels[room_id]["name"] = label["type"]
                rooms_with_labels[room_id]["area"] = label["area"]
                rooms_with_labels[room_id]["label_overlap"] = overlap_amount
    
    # Step 2: Count windows for each room
    for room_id, room in rooms_with_labels.items():
        window_count = 0
        
        # For each window, check if it's on the boundary of this room
        for window in window_detection["merged_pairs"]:
            if is_window_on_room_boundary(window, room):
                window_count += 1
        
        result[room_id] = {
            "name": room["name"] or f"Room {room_id}",
            "area": room.get("area", math.floor(room.get("area", 0) / 144)),  # Convert to square feet if no area from label
            "windows": window_count,
            "boundary": room["boundary"]  # Add the boundary coordinates
        }
    
    return result

def draw_rooms_on_image(image_path, analysis_results, output_path):
    # Read the source image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create a copy to draw on
    img_with_rooms = img.copy()
    
    # Define colors and font
    room_color = (0, 255, 0)  # Green for room boundaries
    text_color = (255, 0, 0)   # Blue for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    for room_id, room_data in analysis_results.items():
        # Get room boundary points
        boundary = np.array(room_data["boundary"], dtype=np.int32)
        
        # Draw the room boundary
        cv2.polylines(img_with_rooms, [boundary], isClosed=True, color=room_color, thickness=2)
        
        # Calculate center position for text (use provided center or calculate from boundary)
        if "center" in room_data:
            center_x, center_y = map(int, room_data["center"])
        else:
            # Calculate centroid if center not provided
            moments = cv2.moments(boundary)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        
        # Prepare text to display
        room_name = room_data["name"]
        window_count = room_data["windows"]
        label_text = f"{room_name} ({window_count} windows)"
        
        # Put the room name and window count
        cv2.putText(img_with_rooms, label_text, 
                   (center_x - 100, center_y), 
                   font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Optionally put area information
        if "area" in room_data:
            area_text = f"{room_data['area']} sq ft"
            cv2.putText(img_with_rooms, area_text,
                       (center_x - 100, center_y + 30),
                       font, font_scale*0.8, text_color, thickness-1, cv2.LINE_AA)
    
    # Save the result
    cv2.imwrite(output_path, img_with_rooms)
    return img_with_rooms

def draw_floor_plan(source_img_path, room_data, output_path="floor_plan_with_windows.png"):
    """
    Draws room boundaries, labels, and windows on a floor plan image.

    Args:
        source_img_path (str): Path to the source floor plan image.
        room_data (dict): Dictionary containing room data (name, windows, boundary points).
        output_path (str, optional): Path to save the output image. Defaults to "floor_plan_with_windows.png".
    """
    # Load the source image
    img = cv2.imread(source_img_path)
    if img is None:
        raise FileNotFoundError(f"Source image not found at {source_img_path}")

    # Generate distinct colors for each room
    def get_distinct_colors(n):
        colors = []
        for i in range(n):
            hue = int(180 * i / n)  # Vary hue for distinct colors
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append([int(c) for c in color])
        return colors

    room_colors = get_distinct_colors(len(room_data))

    # Draw each room's boundary and label
    for room_id, data in room_data.items():
        boundary = np.array(data['boundary'], dtype=np.int32)
        color = room_colors[room_id - 1]  # Get assigned color

        # Draw the polygon boundary
        cv2.polylines(img, [boundary], isClosed=True, color=color, thickness=2)

        # Calculate centroid for placing the label
        centroid = np.mean(boundary, axis=0).astype(int)
        
        # Label the room (name + windows)
        label = f"{data['name']} (Windows: {data['windows']})"
        cv2.putText(img, label, (centroid[0] - 50, centroid[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Mark windows with small circles (approximate positions)
        for _ in range(data['windows']):
            # Pick a random point along the boundary if exact positions are unknown
            window_point = boundary[random.randint(0, len(boundary) - 1)]
            cv2.circle(img, tuple(window_point), 5, (0, 0, 255), -1)  # Red dot

    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Output image saved to {output_path}")

    # # (Optional) Display the image
    # cv2.imshow("Floor Plan with Windows", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def load_ideal_area_and_windows_csv(file_path):
    df = pd.read_csv(file_path)
    area_dict = pd.Series(df['Ideal Area per Person (sq ft)'].values, index=df['Room Type']).to_dict()
    windows_dict = pd.Series(df['Ideal Windows'].values, index=df['Room Type']).to_dict()
    return area_dict, windows_dict

# -------- Space Analysis Function --------
def space_analysis(room_data, ideal_area_per_person, ideal_windows_per_room):
    room_type = room_data.get("Room Type")
    area = room_data.get("area", 0)
    occupants = room_data.get("occupants", 1)
    windows = room_data.get("Windows", 0)

    ideal_area = ideal_area_per_person.get(room_type, 0)
    ideal_windows = ideal_windows_per_room.get(room_type, 2)  # Default to 2 if not found
    total_ideal_area = ideal_area * occupants if ideal_area else 0

    # 1. Space Utilization Score
    space_utilization_score = min(area / total_ideal_area, 1) * 100 if total_ideal_area > 0 else 0

    # 2. Accessibility Score (based on area per occupant)
    area_per_occupant = area / occupants if occupants else 0
    ideal_area_per_occupant = ideal_area
    accessibility_score = min(area_per_occupant / ideal_area_per_occupant, 1) * 100 if ideal_area_per_occupant > 0 else 0

    # 3. Lighting & Ventilation Score (based on ideal windows per room type)
    lighting_ventilation_score = min(windows / ideal_windows, 1) * 100 if ideal_windows > 0 else 0

    # 4. Functional Layout Score
    functional_layout_score = min(area / 100, 1) * 100  # 100 sq ft as baseline for layout quality

    total_score = (
        0.4 * space_utilization_score +
        0.2 * accessibility_score +
        0.2 * lighting_ventilation_score +
        0.2 * functional_layout_score
    )

    return {
        "Room Type": room_type,
        "Space Utilization": space_utilization_score,
        "Accessibility": accessibility_score,
        "Lighting & Ventilation": lighting_ventilation_score,
        "Functional Layout": functional_layout_score,
        "Total Score": total_score
    }

# -------- Bar Chart --------
def plot_combined_bar_chart(room_scores):
    categories = ["Space Utilization", "Accessibility", "Lighting & Ventilation", "Functional Layout"]
    x_labels = [room["Room Type"] for room in room_scores]
    x = np.arange(len(x_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, metric in enumerate(categories):
        values = [room[metric] for room in room_scores]
        ax.bar(x + idx * width, values, width, label=metric)

    ax.set_xlabel("Rooms")
    ax.set_ylabel("Score (out of 100)")
    ax.set_title("Evaluation Metrics per Room")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# -------- Radar Chart --------
def plot_radar_chart(room_scores):
    categories = ["Space Utilization", "Accessibility", "Lighting & Ventilation", "Functional Layout"]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', '*', 'x', 'p', 'h', 'v', '<']

    for i, room in enumerate(room_scores):
        values = [room[c] for c in categories]
        values += values[:1]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(angles, values, linewidth=2, label=room["Room Type"], color=color, marker=marker)
        ax.fill(angles, values, alpha=0.2, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_title("Radar Chart - Evaluation Metrics for Rooms", size=14, y=1.08)
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.show()

def is_standard_room(room_name):
    """Check if room name follows pattern 'Room <number>'"""
    return not bool(re.match(r'^Room\s+\d+$', room_name.strip()))

def analyze_all_rooms(room_data_list, ideal_area_per_person, ideal_windows_per_room):
    all_scores = []
    total_area = 0
    total_score_sum = 0
    total_space_utilization = 0
    total_accessibility = 0
    total_lighting = 0
    total_functional_layout = 0

    for room in room_data_list:
        score = space_analysis(room, ideal_area_per_person, ideal_windows_per_room)
        if(is_standard_room(room.get("Room Type"))):
            total_area += room.get("area", 0)
        total_score_sum += score["Total Score"]
        total_space_utilization += score["Space Utilization"]
        total_accessibility += score["Accessibility"]
        total_lighting += score["Lighting & Ventilation"]
        total_functional_layout += score["Functional Layout"]
        all_scores.append(score)

    avg_score = total_score_sum / len(all_scores)
    avg_space_utilization = total_space_utilization / len(all_scores)
    avg_accessibility = total_accessibility / len(all_scores)
    avg_lighting = total_lighting / len(all_scores)
    avg_functional_layout = total_functional_layout / len(all_scores)

    result = {
        "total_floor_area": total_area,
        "average_scores": {
            "Overall Score": round(avg_score, 2),
            "Space Utilization": round(avg_space_utilization, 2),
            "Accessibility": round(avg_accessibility, 2),
            "Lighting & Ventilation": round(avg_lighting, 2),
            "Functional Layout": round(avg_functional_layout, 2)
        },
        "individual_room_scores": all_scores
    }

    return result

# ---------- Flask Routes ----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    response = {
        'message': 'File uploaded successfully',
        'filename': filename,
        'filepath': filepath
    }
    
    return jsonify(response)

# @app.route('/analyze', methods=['POST'])
# def analyze_floorplan():
#     data = request.json
#     filepath = data.get('filepath')
    
#     # Convert relative path to absolute if needed
#     if filepath.startswith('/static/uploads/'):
#         filepath = os.path.join(base_dir, filepath[1:])  # Remove leading slash
    
#     if not filepath or not os.path.exists(filepath):
#         return jsonify({'error': 'File not found', 'path': filepath}), 404
    
#     result = {}
    
#     # OCR and text clustering
#     try:
#         text_results, text_images = perform_ocr_clustering(filepath)
#         result['text_extraction'] = {
#             'labels': text_results,
#             'images': text_images
#         }
#     except Exception as e:
#         result['text_extraction'] = {'error': str(e)}
    
#     # Room detection
#     # try:
#     #     output_img_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_detection_result.png')
#     #     output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_coordinates.json')
        
#     #     room_data, room_images = detect_and_store_rooms(
#     #         image_path=filepath,
#     #         output_img_path=output_img_path,
#     #         output_json_path=output_json_path,
#     #         wall_thickness=15,
#     #         max_door_width=40
#     #     )
        
#     #     result['room_detection'] = {
#     #         'data': room_data,
#     #         'images': room_images
#     #     }
#     # except Exception as e:
#     #     result['room_detection'] = {'error': str(e)}
    
#     # OCR and text clustering
#     # try:
#     #     text_results, text_images = perform_ocr_clustering(filepath)
#     #     result['text_extraction'] = {
#     #         'labels': text_results,
#     #         'images': text_images
#     #     }
#     # except Exception as e:
#     #     result['text_extraction'] = {'error': str(e)}
    
#     # Room detection
#     try:
#         output_img_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_detection_result.png')
#         output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_coordinates.json')
        
#         room_data, room_images = detect_and_store_rooms(
#             image_path=filepath,
#             output_img_path=output_img_path,
#             output_json_path=output_json_path,
#             wall_thickness=15,
#             max_door_width=40
#         )
        
#         result['room_detection'] = {
#             'data': room_data,
#             'images': room_images
#         }
#     except Exception as e:
#         result['room_detection'] = {'error': str(e)}
    
#     #window detection
#     try: 
#         window_boxes, result_image, main_contour, window_images = detect_windows(
#             filepath, 
#             max_border_distance=20
#         )
        
#         window_data = [list(box) for box in window_boxes]
        
#         # Pair windows
#         paired_windows = pair_windows_by_alignment(
#             window_boxes, 
#             main_contour,
#             max_diff=4, 
#             min_distance=20, 
#             max_distance=200
#         )
        
#         # Draw final result with paired windows
#         paired_image = draw_paired_windows(result_image, window_boxes, paired_windows, main_contour)
#         paired_image_data = save_and_encode_image(paired_image, "paired_windows.png")
        
#         # Add paired image to window_images dictionary
#         window_images['paired'] = paired_image_data
        
#         result['window_detection'] = {
#             'windows': window_data,
#             'paired_windows': [[i, j, alignment, float(distance)] for i, j, alignment, distance in paired_windows],
#             'images': window_images
#         }
#     except Exception as e:
#         result['window_detection'] = {'error': str(e)}
    
#     # Draw all coordinates on a single image
#     try:
#         combined_image_path = draw_all_coordinates(
#             original_image_path=filepath,
#             result_data=result,
#             output_filename=f"combined_analysis_{os.path.basename(filepath)}"
#         )
        
#         # Add the combined image path to the result
#         result['combined_visualization'] = {
#             'image_path': combined_image_path
#         }
#     except Exception as e:
#         result['combined_visualization'] = {'error': str(e)}
    
#     return jsonify(result)

@app.route('/analyze', methods=['POST'])
def analyze_floorplan():
    data = request.json
    filepath = data.get('filepath')
    occupants = data.get('occupants')
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Convert relative path to absolute if needed
    if filepath.startswith('/static/uploads/'):
        filepath = os.path.join(base_dir, filepath[1:])  # Remove leading slash
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found', 'path': filepath}), 404
    
    result = {}
    
    # OCR and text clustering
    try:
        # text_results, text_images = perform_ocr_clustering(filepath)
        # result['text_extraction'] = {
        #     'labels': text_results,
        #     'images': text_images
        # }
        room_data = perform_ocr_clustering(
            image_path=filepath,
            tesseract_path=tesseract_path,
            debug=False,  # Set to True to display debug information and save output image
            draw_boxes=True 
        )
        result['text_extraction'] = {
            'labels': room_data,
        }
        print("Room_Names: ",room_data)
    except Exception as e:
        result['text_extraction'] = {'error': str(e)}
    
    # Room detection
    try:
        output_img_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_detection_result.png')
        output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'room_coordinates.json')
        
        room_data, room_images = detect_and_store_rooms(
            image_path=filepath,
            output_img_path=output_img_path,
            output_json_path=output_json_path,
            wall_thickness=15,
            max_door_width=40
        )
        
        result['room_detection'] = {
            'data': room_data,
            'images': room_images
        }
    except Exception as e:
        result['room_detection'] = {'error': str(e)}
    
    # Window detection
    try: 
        window_boxes, result_image, main_contour, window_images = detect_windows(
            filepath, 
            max_border_distance=20
        )
        
        window_data = [list(map(int, box)) for box in window_boxes]  # Convert to Python int
        
        # Pair windows
        paired_windows = pair_windows_by_alignment(
            window_boxes, 
            main_contour,
            max_diff=4, 
            min_distance=20, 
            max_distance=200
        )
        
        # Convert paired windows to serializable format
        serializable_pairs = []
        for i, j, alignment, distance in paired_windows:
            serializable_pairs.append([
                int(i), 
                int(j), 
                alignment, 
                float(distance)
            ])
        
        # Draw final result with paired windows
        paired_image = draw_paired_windows(result_image, window_boxes, paired_windows, main_contour)
        paired_image_data = save_and_encode_image(paired_image, "paired_windows.png")
        
        # Add paired image to window_images dictionary
        window_images['paired'] = paired_image_data
        
        result['window_detection'] = {
            'windows': window_data,
            'paired_windows': serializable_pairs,
            'images': window_images
        }
        new  = merge_all_paired_windows(result['window_detection'])
        result['window_detection'].update({
            'merged_pairs': new
        })
#         fig, ax = plt.subplots(figsize=(12, 8))

# # Set axis limits to cover all rectangles
#         all_x = [coord[0] for coord in new] + [coord[2] for coord in new]
#         all_y = [coord[1] for coord in new] + [coord[3] for coord in new]
#         ax.set_xlim(min(all_x) - 50, max(all_x) + 50)
#         ax.set_ylim(min(all_y) - 50, max(all_y) + 50)

#         # Flip y-axis to match typical image coordinates (0 at top)
#         ax.invert_yaxis()

#         # Draw each rectangle
#         colors = plt.cm.tab20.colors  # Get a color cycle
#         for i, (x1, y1, x2, y2) in enumerate(new):
#             width = x2 - x1
#             height = y2 - y1
#             rect = patches.Rectangle(
#                 (x1, y1), width, height,
#                 linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none',
#                 label=f'Window {i}'
#             )
#             ax.add_patch(rect)
            
#             # Add label near the rectangle
#             ax.text(x1 + width/2, y1 + height/2, str(i),
#                     ha='center', va='center', color=colors[i % len(colors)])

#         # Add legend and labels
#         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.set_title('Merged Windows Visualization')
#         ax.set_xlabel('X coordinate')
#         ax.set_ylabel('Y coordinate')

#         plt.tight_layout()

#         # Save the figure instead of displaying
#         output_path = "merged_windows_visualization.png"
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {output_path}")

#         # Clear the figure to free memory
#         plt.close()
    except Exception as e:
        result['window_detection'] = {'error': str(e)}
    
    # Draw all coordinates on a single image
    try:
        combined_image_path = draw_all_coordinates(
            original_image_path=filepath,
            result_data=result,
            output_filename=f"combined_analysis_{os.path.basename(filepath)}"
        )
        
        # Add the combined image path to the result
        result['combined_visualization'] = {
            'image_path': combined_image_path
        }
    except Exception as e:
        result['combined_visualization'] = {'error': str(e)}
    
    # Convert numpy types to native Python types before JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    converted_result = convert_numpy_types(result)
    output_dir = os.path.join(".", "static", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    output_json_path = os.path.join(output_dir, "Final_Co_ordinates.json")

    # Write to JSON file
    with open(output_json_path, "w") as f:
        json.dump(converted_result, f, indent=4)
    
    final = jsonify(converted_result)

    results = analyze_rooms_and_windows(converted_result["room_detection"], converted_result["window_detection"], converted_result["text_extraction"])
    print(results)
    draw_floor_plan(
        source_img_path=filepath,
        room_data=results,
        output_path="custom_output.png"
    )
    #default_occupants = 4
    formatted_rooms = []
    for room in results.values():
        formatted_room = {
            "Room Type": room["name"],
            "area": room["area"],
            "Units": "sq ft",
            "Windows": room["windows"],
            "coordinates": room["boundary"],
            "occupants": occupants
        }
        formatted_rooms.append(formatted_room)

    csv_path = "ideal_room_area_per_person_with_windows.csv"
    ideal_area_per_person, ideal_windows_per_room = load_ideal_area_and_windows_csv(csv_path)
    
    final_1 = analyze_all_rooms(formatted_rooms,ideal_area_per_person,ideal_windows_per_room)
    final_1['combined_visualization'] = {
            'image_path':  "custom_output.png"
    }
    return final_1

@app.route('/output/<filename>')
def output_image(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)