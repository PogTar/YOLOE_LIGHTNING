import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

# --- DIRECTORY CONFIGURATION ---
INPUT_IMAGES_DIR = 'datasets/generated_dataset/valid/images'
INPUT_LABELS_DIR = "/root/repos/yoloe_lightning/datasets/generated_dataset/valid/labels"
INPUT_MASKS_DIR = 'datasets/generated_dataset/valid/pseudo_masks'
OUTPUT_JSON_DIR = 'datasets/generated_dataset/valid/output_json'

# --- CLASS MAPPING ---
CLASS_MAP = {
0:'triangle', 1:'rectangle', 2:'circle'
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

def yolo_to_pixel(x_c, y_c, w, h, img_w, img_h):
    """Converts YOLO normalized coordinates to absolute [x_min, y_min] and [x_max, y_max]"""
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    # Rounding to integers as per your JSON example
    return [int(x1), int(y1)], [int(x2), int(y2)]


def extract_polygons_from_mask(mask_path, class_map):
    """Reads a segmentation mask and returns a list of polygon dictionaries."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    polygons = []
    unique_classes = np.unique(mask)
    
    for class_id in unique_classes:
        # Skip unmapped classes
        if  class_id not in class_map:
            print(f"unmapped class,class_id is {class_id}")
            continue
            
        label = class_map[class_id]
        
        # Isolate the specific class
        binary_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)
        
        # Extract contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # A valid polygon needs at least 3 points
            if len(contour) < 3:
                continue
                
            # Flatten to [[x, y], [x, y]] and convert to standard Python int for JSON serialization
            points = [[int(pt[0][0]), int(pt[0][1])] for pt in contour]
            
            shape = {
                "description": "",
                "label": label,
                "points": points,
                "pose": "Unspecified",
                "re_id": None,
                "type": "polygon",
                "uuid": "" 
            }
            polygons.append(shape)
            
    return polygons

import os
import json
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# Set to True to save absolute pixel coordinates. 
# Set to False to keep coordinates normalized (0.0 to 1.0).
CONVERT_TO_PIXELS = True 

if __name__ == "__main__":

    for label_file in tqdm(os.listdir(INPUT_LABELS_DIR)):
        if not label_file.endswith('.txt'):
            continue
        
        base_name = os.path.splitext(label_file)[0]
        
        # Define the formats you expect in your dataset
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        img_path = None

        # Loop through and check which one actually exists
        for ext in valid_extensions:
            temp_path = os.path.join(INPUT_IMAGES_DIR, f"{base_name}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                break # Stop looking once we find it!

        if img_path is None:
            print(f"⚠️ Warning: No image found for {base_name}")
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
            depth = len(img.getbands()) # Usually 3 for RGB

        shapes = []
        
        # Read YOLO text file
        with open(os.path.join(INPUT_LABELS_DIR, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                
                # --- COORDINATE CONVERSION LOGIC ---
                if CONVERT_TO_PIXELS:
                    # Convert to absolute pixel coordinates
                    p1, p2 = yolo_to_pixel(x_c, y_c, w, h, width, height)
                else:
                    # Keep normalized (0.0 to 1.0) but convert from center to min/max corners
                    x_min = x_c - (w / 2.0)
                    y_min = y_c - (h / 2.0)
                    x_max = x_c + (w / 2.0)
                    y_max = y_c + (h / 2.0)
                    
                    # Ensure coordinates stay strictly within 0-1 bounds
                    p1 = [max(0.0, x_min), max(0.0, y_min)]
                    p2 = [min(1.0, x_max), min(1.0, y_max)]
                
                # Create shape dictionary
                shape = {
                    "description": "",
                    "label": CLASS_MAP.get(class_id, "unknown"),
                    "points": [p1, p2],
                    "pose": "Unspecified",
                    "re_id": None,
                    "type": "rectangle",
                    "uuid": ""  # Set to empty as requested
                }
                shapes.append(shape)

       # 3. Extract Segmentation Masks (Polygons)
        # Assuming masks are named exactly the same but with .png extension
        mask_path = os.path.join(INPUT_MASKS_DIR, f"{base_name}.png")
        if os.path.exists(mask_path):
            polygon_shapes = extract_polygons_from_mask(mask_path, CLASS_MAP)
            shapes.extend(polygon_shapes)  # Add polygons to the existing shapes list
        else:
            # Optional: warn if a mask is missing, but still generate JSON for YOLO boxes
            tqdm.write(f"Notice: No mask found for '{base_name}.png'. Only YOLO boxes extracted.")

        # 4. Build Final JSON Structure
        output_data = {
            "flags": {
                "bad_image": False,
                "dark": False,
                "difficult": False,
                "later": False,
                "long_distance": False
            },
            "metadata": {
                "depth": depth,
                "height": height,
                "width": width
            },
            "shapes": shapes,  # Now contains BOTH rectangles and polygons
            "tags": {
                "angled": True,
                "ground": True,
                "overhead": False
            }
        }
        # Save to JSON file
        json_path = os.path.join(OUTPUT_JSON_DIR, f"{base_name}.json")
        with open(json_path, 'w') as out_f:
            json.dump(output_data, out_f, indent=4)

    print(f"✅ Conversion complete! All JSON files have been saved to the '{OUTPUT_JSON_DIR}' folder.")