# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Utility to display image
# def show_image(img, title, cmap='gray'):
#     plt.figure(figsize=(10, 10))
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         cmap = None
#     plt.imshow(img, cmap=cmap)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# # Step 1: Load grayscale image
# image_path = '../static/uploads/floor_plan.jpg'
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# show_image(img, "Original Grayscale Image")

# # Step 2: Invert image
# img_inv = cv2.bitwise_not(img)
# show_image(img_inv, "Inverted Image")

# # Step 3: Adaptive thresholding
# thresh = cv2.adaptiveThreshold(img_inv, 255, 
#                                cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                cv2.THRESH_BINARY, 15, -2)
# show_image(thresh, "Adaptive Thresholded Image")

# # Step 4: Morphological closing to seal gaps in walls
# kernel = np.ones((5, 5), np.uint8)
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
# show_image(closed, "After Morphological Closing")

# # Step 5: Invert to get black walls, white rooms
# binary = cv2.bitwise_not(closed)
# show_image(binary, "Binary for Room Segmentation")

# # Step 6: Connected components analysis
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# # Step 7: Draw valid bounding boxes
# # Initialize an empty dictionary to store room coordinates
# # Initialize an empty dictionary to store room coordinates
# room_coordinates = {}

# # Step 7: Draw valid bounding boxes and store coordinates
# output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
# room_id = 0

# for i in range(1, num_labels):  # Skip background
#     x, y, w, h, area = stats[i]
#     aspect_ratio = w / h if h != 0 else 0

#     # Filter small or strange regions
#     if area > 7000 and w > 40 and h > 40 and 0.3 < aspect_ratio < 3.5:
#         room_id += 1
#         # Draw bounding box
#         cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
#         # Store the coordinates of the two corners in the dictionary
#         top_left = (x, y)
#         bottom_right = (x + w, y + h)
#         room_coordinates[room_id] = (top_left, bottom_right)

# # Output the dictionary with room coordinates
# print("Room Coordinates Dictionary:", room_coordinates)

# # Show the image with bounding boxes
# show_image(output, "Filtered Room Boxes (Cleaned Output)")


# show_image(output, "Filtered Room Boxes (Cleaned Output)")


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Utility to display image
def show_image(img, title, cmap='gray'):
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = None
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Step 1: Load grayscale image
image_path = '../static/uploads/floor_plan1.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
show_image(img, "Original Grayscale Image")

# Step 2: Invert image
img_inv = cv2.bitwise_not(img)
show_image(img_inv, "Inverted Image")

# Step 3: Adaptive thresholding
thresh = cv2.adaptiveThreshold(img_inv, 255, 
                               cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 15, -2)
show_image(thresh, "Adaptive Thresholded Image")

# Step 4: Morphological closing to seal gaps in walls
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
show_image(closed, "After Morphological Closing")

# Step 5: Invert to get black walls, white rooms
binary = cv2.bitwise_not(closed)
show_image(binary, "Binary for Room Segmentation")

# Step 6: Connected components analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Step 7: Draw valid bounding boxes
# Initialize an empty dictionary to store room coordinates
# Initialize an empty dictionary to store room coordinates
room_coordinates = {}

# Step 7: Draw valid bounding boxes and store coordinates
output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
room_id = 0

for i in range(1, num_labels):  # Skip background
    x, y, w, h, area = stats[i]
    aspect_ratio = w / h if h != 0 else 0

    # Filter small or strange regions
    if area > 7000 and w > 40 and h > 40 and 0.3 < aspect_ratio < 3.5:
        room_id += 1
        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Store the coordinates of the two corners in the dictionary
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        room_coordinates[room_id] = (top_left, bottom_right)

# Output the dictionary with room coordinates
#print("Room Coordinates Dictionary:", room_coordinates)

# Show the image with bounding boxes
show_image(output, "Filtered Room Boxes (Cleaned Output)")

# Step 8: Remove the first box
if 1 in room_coordinates:
    del room_coordinates[1]

# Step 9: Filter out boxes completely inside another
final_room_coordinates = {}

def is_inside(inner, outer):
    (ix1, iy1), (ix2, iy2) = inner
    (ox1, oy1), (ox2, oy2) = outer
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

# Reassign room_id starting from 1
new_room_id = 1
coords_list = list(room_coordinates.values())

for i, box in enumerate(coords_list):
    inside_another = False
    for j, other_box in enumerate(coords_list):
        if i == j:
            continue
        if is_inside(box, other_box):
            inside_another = True
            break
    if not inside_another:
        final_room_coordinates[new_room_id] = box
        new_room_id += 1

# Step 10: Draw filtered boxes
filtered_output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

for room_id, (top_left, bottom_right) in final_room_coordinates.items():
    cv2.rectangle(filtered_output, top_left, bottom_right, (0, 255, 0), 2)

# Output final coordinates and show image
print("Filtered Room Coordinates (No Nested Boxes):", final_room_coordinates)
show_image(filtered_output, "Final Room Boxes (No Nested Boxes)")



