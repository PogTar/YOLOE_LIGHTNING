import json
from pathlib import Path
from tqdm import tqdm
import os


def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Converts corner coordinates to YOLO format.
    
    Args:
        x1, y1: Top-left coordinates
        x2, y2: Bottom-right coordinates
        img_w, img_h: Image width and height
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized
    """
    
    # 1. Calculate center, width, and height in pixels
    w_box = x2 - x1
    h_box = y2 - y1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # 2. Normalize by image dimensions
    x_n = x_center / img_w
    y_n = y_center / img_h
    w_n = w_box / img_w
    h_n = h_box / img_h

    # 3. Return formatted to 6 decimal places (standard for YOLO)
    return float(f"{x_n:.6f}"), float(f"{y_n:.6f}"), float(f"{w_n:.6f}"), float(f"{h_n:.6f}")

# def collect_by_image_angle():
#     for labels_structure in json_labels:
#         try:
#             with open(labels_structure) as file:
#                 data = json.load(file)
#                 for angle_type in data["tags"]:
#                     if data["tags"][angle_type] == True:
#                         try:
#                             # Copy the image  file from common to categorized by angle folder
#                             # If the destination is an existing directory, the file is copied into that directory
#                             source_file = "/home/taron-poghosyan/Desktop/YOLOE/yoloe/skadi_images_in_yolo_format/images/" + labels_structure.stem + '.jpg' ###we copy appropriate jpg image  of json file from skadi_images_in_yolo_format
#                             destination_file = "/home/taron-poghosyan/Desktop/YOLOE/yoloe/skadi_images_in_yolo_format/categorized_by_angle/" + f"{angle_type}" + "/images/" +  labels_structure.stem + '.jpg'
#                             shutil.copy2(source_file, destination_file)
#                         except shutil.SameFileError:
#                             print("Source and destination represent the same file.")
#                         except PermissionError:
#                             print("Permission denied.")
#                         except IsADirectoryError:
#                             print("Destination is a directory, but a file name was expected.")
#                         except FileNotFoundError:
#                             print("Source file or destination directory not found.")
#                         except Exception as e:
#                             print(f"An error occurred: {e}")
#                         try:            
#                             #copy appropriate label txt file from common labels to categorized labels
#                             source_file = "/home/taron-poghosyan/Desktop/YOLOE/yoloe/skadi_images_in_yolo_format/labels/" + labels_structure.stem + '.txt' 
#                             destination_file = "/home/taron-poghosyan/Desktop/YOLOE/yoloe/skadi_images_in_yolo_format/categorized_by_angle/" + f"{angle_type}"+"/labels/" +  labels_structure.stem + '.txt' 
#                             shutil.copy2(source_file, destination_file)
#                         except shutil.SameFileError:
#                             print("Source and destination represent the same file.")
#                         except PermissionError:
#                             print("Permission denied.")
#                         except IsADirectoryError:
#                             print("Destination is a directory, but a file name was expected.")
#                         except FileNotFoundError:
#                             print("Source file or destination directory not found.")
#                         except Exception as e:
#                             print(f"An error occurred: {e}")                            




#         except FileNotFoundError:
#             print("Error: The file 'data.json' was not found.")
#         except json.JSONDecodeError as e:
#             print(f"Error: Failed to decode JSON from the file: {e}")          


def convert_json_to_yolo(json_labels, output_dir, initial_mapping=None):
    """
    Converts JSON labels to YOLO format, dynamically generating class IDs for new classes.
    
    Args:
        json_labels (list): List of Path objects pointing to JSON files.
        output_dir (str/Path): Directory where YOLO .txt files will be saved.
        initial_mapping (dict, optional): Starting dictionary for class names to IDs.
        
    Returns:
        dict: The final mapping of class names to IDs.
    """
    # Initialize the mapping dictionary
    classname2id = initial_mapping if initial_mapping is not None else {}
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for labels_structure in tqdm(json_labels):
        try:
            with open(labels_structure, 'r') as file:
                data = json.load(file)

            height = data["metadata"]["height"]
            width = data["metadata"]["width"]
            
            yolo_labels = []

            # Iterate through shapes to dynamically update classes
            for shapes in data["shapes"]:
                class_name = shapes["label"]
                
                # If the class is not in the dictionary, add it with a new ID
                if class_name not in classname2id:
                    # The new ID is simply the current number of classes mapped
                    classname2id[class_name] = len(classname2id)
                
                class_id = classname2id[class_name]
                
                # Assuming convert_to_yolo is defined elsewhere in your code
                yolo_coords = convert_to_yolo(
                    *shapes["points"][0], 
                    *shapes["points"][1], 
                    width, 
                    height
                )
                
                yolo_labels.append((class_id, yolo_coords))
            
            # Create the .txt file using pathlib
            file_name = Path(output_dir) / f"{labels_structure.stem}.txt"
            
            with open(file_name, "w") as txt_file:
                for item in yolo_labels:
                    txt_file.write(f"{item[0]} {item[1][0]} {item[1][1]} {item[1][2]} {item[1][3]}\n")

        except FileNotFoundError:
            print(f"Error: The file '{labels_structure}' was not found.")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from the file '{labels_structure}': {e}")
        except KeyError as e:
            print(f"Error: Missing expected key {e} in the file '{labels_structure}'.")

    # Return the final mapping once all files are processed
    return classname2id   


def remove_unlabeled_images(images_dir, labels_dir, label_ext=".json", dry_run=True):
    """
    Iterates through an image directory and deletes images that lack a corresponding label file.
    
    Args:
        images_dir (str/Path): Path to the folder containing images.
        labels_dir (str/Path): Path to the folder containing label files.
        label_ext (str): The extension of your label files (default is ".json").
        dry_run (bool): If True, only prints what would be deleted. If False, actually deletes.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Common image extensions to ensure we don't accidentally delete system files (like .DS_Store)
    valid_img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Check if directories exist
    if not images_path.exists():
        print(f"Error: Image directory '{images_dir}' not found.")
        return
    if not labels_path.exists():
        print(f"Error: Label directory '{labels_dir}' not found.")
        return

    unmatched_count = 0
    print(f"--- Starting Scan | Dry Run: {dry_run} ---")

    for img_file in tqdm(images_path.iterdir()):
        # Ensure it's a file and an actual image
        if img_file.is_file() and img_file.suffix.lower() in valid_img_extensions:
            
            # Construct the path to where the label file SHOULD be
            expected_label = labels_path / f"{img_file.stem}{label_ext}"
            
            # Check if the expected label file exists
            if not expected_label.exists():
                unmatched_count += 1  # Tally the unmatched file
                
                if dry_run:
                    print(f"[DRY RUN] Would delete: {img_file.name} (Missing {expected_label.name})")
                else:
                    try:
                        img_file.unlink() # This permanently deletes the file
                        print(f"Deleted: {img_file.name}")
                    except Exception as e:
                        print(f"Failed to delete {img_file.name}. Error: {e}")

    # Final summary
    if dry_run:
        print("\n--- Dry Run Complete ---")
        print(f"Found {unmatched_count} unmatched image(s) that would be deleted.")
        print("Change 'dry_run=False' to execute deletion.")
    else:
        print("\n--- Cleanup Complete ---")
        print(f"Successfully deleted {unmatched_count} image(s).")


# ####I had problem with downloading images ,Images are more than downloaded labels so I keep only those images whose labels are already downloaded and it is enough for me
# def copy_yolo_images():
#     # Define your paths
#     base_yolo_dir = Path("datasets/skadi_in_yolo_format")
#     labels_dir = base_yolo_dir / "labels"
#     dst_images_dir = base_yolo_dir / "images"
    
#     src_images_dir = base_yolo_dir / "skadi_images/images"

#     # 1. Create the destination images folder (if it doesn't already exist)
#     dst_images_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Destination folder ready at: {dst_images_dir}")

#     # 2. Map all source images by their filename without the extension
#     # This ensures we find the image whether it is a .jpg, .png, etc.
#     if not src_images_dir.exists():
#         print(f"Error: Source images directory not found at {src_images_dir}")
#         return

#     src_images_map = {f.stem: f for f in src_images_dir.iterdir() if f.is_file()}

#     # 3. Iterate through label files and copy the matching images
#     copied_count = 0
#     missing_count = 0

#     for label_file in labels_dir.glob("*.txt"):
#         image_name_base = label_file.stem  # Gets the filename without '.txt'
        
#         if image_name_base in src_images_map:
#             src_image_path = src_images_map[image_name_base]
#             dst_image_path = dst_images_dir / src_image_path.name
            
#             # copy2 preserves file metadata (like timestamps)
#             shutil.copy2(src_image_path, dst_image_path)
#             copied_count += 1
#         else:
#             print(f"Warning: No matching image found for label '{label_file.name}'")
#             missing_count += 1

#     # 4. Final summary
#     print("-" * 30)
#     print("Copying Complete!")
#     print(f"Successfully copied: {copied_count} images")
#     print(f"Missing images: {missing_count}")               

if __name__ == "__main__":
    # json_labels_path = Path("/mnt/ceph-nvme/datasets/sam3_data/labels_SAM3")
    # json_labels = list(json_labels_path.glob("*.json"))
    # final_mapping = convert_json_to_yolo(json_labels=json_labels,output_dir="datasets/sam3/labels")
    # print(final_mapping)
    remove_unlabeled_images(images_dir="datasets/sam3/images",labels_dir="datasets/sam3/labels",label_ext=".txt",dry_run=False)