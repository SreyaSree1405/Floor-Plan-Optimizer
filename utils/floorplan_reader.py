import re
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_line(line):
    line = line.strip()
    if not line or len(line) < 2:
        return None
    if len(re.findall(r'[a-zA-Z]', line)) < 2 and not re.search(r'\d', line):
        return None
    return line

def extract_room_data_from_image(image_path):
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    print("🔍 Raw OCR Output:\n", raw_text)

    lines = [clean_line(line) for line in raw_text.split('\n')]
    lines = [line for line in lines if line]  # Remove None values

    results = []
    buffer = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if current line ends with "sq" and next line is "ft"
        if re.search(r'\d+\s*sq$', line.lower()) and i + 1 < len(lines) and lines[i + 1].lower() == 'ft':
            full_size = line + " ft"
            room_name = " ".join(buffer).strip()
            results.append((room_name, full_size))
            buffer = []
            i += 2
            continue

        # Check if this line contains full "sq ft"
        if re.search(r'\d+\s*sq\s*ft', line.lower()):
            room_name = " ".join(buffer).strip()
            results.append((room_name, line))
            buffer = []
        else:
            buffer.append(line)

        i += 1

    print("\n📋 Extracted Room Names and Sizes:")
    for name, size in results:
        print(f"- {name} => {size}")

# Run the function
image_path = '../static/uploads/floor_plan1.jpg'
extract_room_data_from_image(image_path)