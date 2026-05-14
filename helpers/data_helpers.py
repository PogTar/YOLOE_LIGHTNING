import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def calculate_class_statistics(labels_dir, class_mapping=None):
    """
    Iterates through YOLO format .txt label files and counts instances per class.
    
    Args:
        labels_dir (str/Path): Path to the folder containing YOLO .txt label files.
        class_mapping (dict, optional): Dictionary mapping class names to IDs (e.g., {"car": 0, "truck": 1}).
                                        If provided, the output will show class names.
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"Error: Label directory '{labels_dir}' not found.")
        return

    # Counter to keep track of instance counts per class ID
    class_counts = Counter()
    total_instances = 0
    total_files_read = 0

    print("--- Calculating Dataset Statistics ---")

    for txt_file in tqdm(labels_path.glob("*.txt")):
        try:
            with open(txt_file, "r") as file:
                lines = file.readlines()
                total_files_read += 1
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    
                    # YOLO format: class_id x_center y_center width height
                    # We only need the first element
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_instances += 1
                    
        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")

    # Display the results
    print("\n--- Statistics Summary ---")
    print(f"Total Label Files Read: {total_files_read}")
    print(f"Total Instances (Bounding Boxes): {total_instances}\n")
    
    # Invert the mapping to easily look up names by ID, if provided
    id_to_name = {}
    if class_mapping:
        id_to_name = {v: k for k, v in class_mapping.items()}

    # Print the counts sorted by Class ID
    print(f"{'Class ID':<10} | {'Class Name':<20} | {'Instance Count'}")
    print("-" * 55)
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        # Get the name if mapping exists, otherwise default to "Unknown"
        class_name = id_to_name.get(class_id, "Unknown") 
        print(f"{class_id:<10} | {class_name:<20} | {count}")
        

def get_yolo_target_indices(val_label_dir, target_class_id='2', offset=0):
    """
    Scans YOLO txt files to find indices of images containing the target class.
    'offset' is used if this dataset is concatenated after a training dataset.
    """
    target_indices = []
    # Assuming labels are named 0001.txt, 0002.txt, etc., and sorted
    label_files = sorted(os.listdir(val_label_dir))
    
    for i, file_name in enumerate(label_files):
        if not file_name.endswith('.txt'):
            continue
            
        file_path = os.path.join(val_label_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                # YOLO format: <class_id> <x> <y> <w> <h>
                if line.strip().split(' ')[0] == str(target_class_id):
                    target_indices.append(i + offset)
                    break # Found the class, move to next file
                    
    return target_indices        


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    LABELS_FOLDER = 'datasets/sam3/labels/'
    
    # This is the mapping your first script generated
    my_mapping = {
    'building': 0, 
    'car': 1, 
    'truck': 2, 
    'aircraft': 3, 
    'ifv': 4, 
    'anti_air_vehicle': 5, 
    'person': 6, 
    'personnel': 7, 
    'heavy_equipment': 8, 
    'tent': 9, 
    'covered_object': 10, 
    'pickup': 11, 
    'van_minivan': 12, 
    'mlrs': 13, 
    'self_propelled_howitzer': 14, 
    'armored': 15, 
    'gun_howitzer': 16, 
    'destroyed_object': 17, 
    'armored_car': 18, 
    'apc': 19, 
    'helicopter': 20, 
    'shelter': 21, 
    'missile_system': 22
    }
    
    calculate_class_statistics(LABELS_FOLDER, class_mapping=my_mapping)