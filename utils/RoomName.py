# import cv2
# import pytesseract
# import numpy as np
# from sklearn.cluster import DBSCAN
# import re
# import unicodedata

# # Load image
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# image_path = '../static/uploads/floor_plan.jpg'
# image = cv2.imread(image_path)
# if image is None:
#     print(f"Error: Could not load image from {image_path}")
#     exit()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# # OCR config
# custom_config = r'--oem 3 --psm 6'
# data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

# # Extract bounding boxes
# boxes = []
# for i in range(len(data['text'])):
#     if data['text'][i].strip() != "":
#         x = data['left'][i]
#         y = data['top'][i]
#         w = data['width'][i]
#         h = data['height'][i]
#         boxes.append([x, y, x + w, y + h])  # x1, y1, x2, y2

# boxes_np = np.array(boxes)
# room_data = []

# def normalize(text):
#     text = unicodedata.normalize('NFKD', text)
#     return ''.join(c for c in text if c.isprintable()).strip()

# # Cluster boxes
# if len(boxes_np) > 0:
#     centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in boxes_np])
#     widths = boxes_np[:, 2] - boxes_np[:, 0]
#     heights = boxes_np[:, 3] - boxes_np[:, 1]
#     avg_size = int((np.mean(widths) + np.mean(heights)) / 2)
#     eps = max(30, avg_size * 1.2)

#     clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
#     labels = clustering.labels_

#     for label in set(labels):
#         cluster_boxes = boxes_np[labels == label]
#         x1 = np.min(cluster_boxes[:, 0]) - 10
#         y1 = np.min(cluster_boxes[:, 1]) - 10
#         x2 = np.max(cluster_boxes[:, 2]) + 10
#         y2 = np.max(cluster_boxes[:, 3]) + 10

#         # Clamp to image bounds
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)

#         roi = gray[y1:y2, x1:x2]
#         roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#         roi = cv2.GaussianBlur(roi, (3, 3), 0)

#         text = pytesseract.image_to_string(roi, config='--psm 6').strip()
#         text = normalize(text)

#         if not text or re.fullmatch(r"[\W_]+", text) or (len(text) <= 2 and not any(c.isdigit() for c in text)):
#             continue

#         # Match pattern like: "Bedroom 1 174 sq ft"
#         match = re.search(r"(?P<name>[A-Za-z/ ]+)(\s*\d*)?\s*(?P<area>\d+)\s*(?P<unit>sq\s*ft|m2|m²|m\?)", text, re.IGNORECASE)
#         if match:
#             name = match.group("name").strip().title()
#             area = int(match.group("area"))
#             unit = match.group("unit").replace("?", "2").lower()

#             room_entry = {
#                 "area": area,
#                 "unit": unit,
#                 "coordinates": [
#                     [int(x1), int(y1)],
#                     [int(x2), int(y1)],
#                     [int(x2), int(y2)],
#                     [int(x1), int(y2)]
#                 ]
#             }

#             room_data.append((name, room_entry))

# # Convert to dictionary format
# room_dict = {}
# for name, data in room_data:
#     if name in room_dict:
#         # Optional: merge or differentiate duplicates
#         name = f"{name}_{len(room_dict)+1}"
#     room_dict[name] = data

# # Print result
# import json
# print(json.dumps(room_dict, indent=2))

# import cv2
# import pytesseract
# import numpy as np
# from sklearn.cluster import DBSCAN
# import re
# import unicodedata
# import os

# # Set your Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load and preprocess image
# base_dir = os.path.dirname(__file__)  # Path to where Test.py is located
# image_path = os.path.join(base_dir, '..', 'static', 'uploads', 'floor_plan.jpg')
# image_path = os.path.abspath(image_path)

# image = cv2.imread(image_path)
# if image is None:
#     print(f"Error: Could not load image from {image_path}")
#     exit()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# # OCR config
# custom_config = r'--oem 3 --psm 6'
# data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

# # Extract bounding boxes for non-empty text entries
# boxes = []
# for i in range(len(data['text'])):
#     if data['text'][i].strip() != "":
#         x = data['left'][i]
#         y = data['top'][i]
#         w = data['width'][i]
#         h = data['height'][i]
#         boxes.append([x, y, x + w, y + h])  # (x1, y1, x2, y2)

# boxes_np = np.array(boxes)
# extracted_texts = []

# # Cluster boxes using DBSCAN
# if len(boxes_np) > 0:
#     centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in boxes_np])
#     widths = boxes_np[:, 2] - boxes_np[:, 0]
#     heights = boxes_np[:, 3] - boxes_np[:, 1]
#     avg_size = int((np.mean(widths) + np.mean(heights)) / 2)
#     eps = max(30, avg_size * 1.2)

#     clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
#     labels = clustering.labels_

#     for label in set(labels):
#         cluster_boxes = boxes_np[labels == label]
#         x1 = np.min(cluster_boxes[:, 0])
#         y1 = np.min(cluster_boxes[:, 1])
#         x2 = np.max(cluster_boxes[:, 2])
#         y2 = np.max(cluster_boxes[:, 3])

#         # Expand box slightly
#         padding = 10
#         x1_p = max(0, x1 - padding)
#         y1_p = max(0, y1 - padding)
#         x2_p = min(gray.shape[1], x2 + padding)
#         y2_p = min(gray.shape[0], y2 + padding)

#         # Crop and preprocess the expanded box
#         roi = gray[y1_p:y2_p, x1_p:x2_p]
#         roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#         roi = cv2.GaussianBlur(roi, (3, 3), 0)

#         # OCR on the cropped region
#         text = pytesseract.image_to_string(roi, config='--psm 6').strip()
#         if text:
#             extracted_texts.append(text)
#             # Draw rectangle and label
#             cv2.rectangle(image, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2)
#             cv2.putText(image, text, (x1_p, y1_p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

# # Function to normalize and clean extracted text
# def normalize(text):
#     text = unicodedata.normalize('NFKD', text)  # Normalize unicode
#     text = ''.join([c for c in text if c.isprintable()])  # Remove non-printable chars
#     return text.strip()

# def clean_text(extracted_texts):
#     cleaned_texts = []

#     for raw_text in extracted_texts:
#         text = normalize(raw_text)

#         # Remove junk
#         if re.fullmatch(r"[\W_]+", text):
#             continue
#         if len(text) <= 2 and not any(char.isdigit() for char in text):
#             continue

#         # Keep if it contains room and area
#         if re.search(r"\d+\s?(sq\s?ft|sqft|m2|m²|m\?)", text, re.IGNORECASE):
#             cleaned_texts.append(text)

#     return cleaned_texts

# # Run cleaner
# final_cleaned = clean_text(extracted_texts)

# # Output the cleaned text
# print("Cleaned Extracted Room Labels:")
# for t in final_cleaned:
#     print("-", t)

# # Save and show result
# cv2.imwrite('output_with_text_clusters.png', image)
# cv2.imshow("Merged Text Clusters", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import pytesseract
import numpy as np
from sklearn.cluster import DBSCAN
import re
import unicodedata
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Set your Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load and preprocess image
base_dir = os.path.dirname(__file__)  # Path to where Test.py is located
image_path = os.path.join(base_dir, '..', 'static', 'uploads', 'floor_plan.jpg')
image_path = os.path.abspath(image_path)

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# OCR config
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

        # Expand box slightly
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
            # Draw rectangle and label
            cv2.rectangle(image, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2)
            cv2.putText(image, text, (x1_p, y1_p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

def clean_and_normalize_text(text):
    """Normalize and clean OCR-extracted text"""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if c.isprintable()).strip()
    return re.sub(r'\s+', ' ', text)

def extract_room_data(text_items):
    """
    Extract structured room information with proper area parsing.
    Handles multiple instances of the same room type (e.g., multiple bedrooms).
    
    Args:
        text_items: List of (text, coordinates) tuples
        
    Returns:
        Dictionary with room data organized by room type and ID
    """
    import re
    from collections import defaultdict
    
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
        text = clean_and_normalize_text(text)
        
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
                'raw_text': text,
                'type': name  # Adding type field for compatibility with draw_all_coordinates
            }
            
            # Store by room ID
            rooms[room_id] = room_data
            
            # Add to the room type grouping
            room_types[name].append(room_data)
    
    # Add a list of all rooms for compatibility with draw_all_coordinates
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

# Run cleaner
print("Before Cleaning: ",extracted_texts)  
final_cleaned = extract_room_data(extracted_texts)

# Output the cleaned text along with coordinatesprint("Cleaned Extracted Room Labels:")
if 'by_id' in final_cleaned:  # Check if rooms were found
    for room_id, room_data in final_cleaned['by_id'].items():
        print(f"{room_id}: Area: {room_data['area']} {room_data['unit']}, Coordinates: {room_data['coordinates']}")
else:
    print("No room labels were extracted from the image.")

# Save and show result
cv2.imwrite('output_with_text_clusters.png', image)

# Display the image using Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with Matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Hide axis
plt.show()
