import cv2
import numpy as np

def detect_rooms(floor_plan_path, output_path):
    # Load the floor plan image
    img = cv2.imread(floor_plan_path)
    if img is None:
        print("Error: Could not load image")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image (walls will be white)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to thicken walls and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours (external only)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image (white background)
    output = np.ones_like(img) * 255
    
    # Draw each room with different colors
    for i, contour in enumerate(contours):
        # Approximate the contour to simplify it
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw the room
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.drawContours(output, [approx], -1, color, 2)
        
        # Optionally, fill the room with semi-transparent color
        overlay = output.copy()
        cv2.fillPoly(overlay, [approx], color)
        alpha = 0.3  # Transparency factor
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
    
    # Save the output
    cv2.imwrite(output_path, output)
    print(f"Room detection complete. Output saved to {output_path}")

# Example usage
detect_rooms('../static/uploads/floor_plan.jpg', 'room_detection_output.png')