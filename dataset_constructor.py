from torch.utils.data import DataLoader,Dataset
from pathlib import Path
from PIL import Image
import torch
import os
import numpy as np
import albumentations as A
import cv2


class YOLODataset(Dataset):
    """
    WARNING!!!!!!
    __getitiem__() method expect that labels are in yolo format(normalized).
    So 
    1.If labels are in another format and/or not normalized
    2.If Width of image is NOT equal to height

    then not use Resize or any transform or use but don't forget about appropriate label transforms also
    
    
    """
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        self.safe_resize = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(
        min_height=512, 
        min_width=512, 
        border_mode=cv2.BORDER_CONSTANT, 
        fill=(114, 114, 114)
       )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],clip=True))
        
        # 1. Robust Image Listing (Support jpg, png, jpeg, bmp)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.img_paths = []
        for ext in extensions:
            self.img_paths.extend(list(self.img_dir.glob(ext)))
            
        # Sort to ensure same order on every run (crucial for valid/test sets)
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Image (Keep your excellent fallback logic)
        img_path = self.img_paths[idx]
        try:
            # Load with PIL, but immediately convert to a NumPy array for Albumentations
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil) 
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # 2. Handle Labels
        label_name = img_path.stem + '.txt'
        label_path = self.label_dir / label_name

        labels = np.zeros((0, 5)) # Default: Empty [N, 5] array
        
        if label_path.exists() and os.path.getsize(label_path) > 0:
            try:
                labels = np.loadtxt(label_path, ndmin=2) 
                if labels.shape[1] != 5:
                    print(f"Warning: malformed label in {label_path}")
                    labels = np.zeros((0, 5))
            except Exception as e:
                print(f"Error reading label {label_path}: {e}")

        # Extract classes and boxes for Albumentations
        if len(labels) > 0:
            class_labels = labels[:, 0].tolist()
            bboxes = labels[:, 1:].tolist()
        else:
            class_labels = []
            bboxes = []

        # 3. Apply Safe Resizing & Padding (Letterboxing)
        # This resizes the image to 512x512 AND corrects the YOLO coordinates
        transformed = self.safe_resize(
            image=img_np, 
            bboxes=bboxes, 
            class_labels=class_labels
        )
        
        img_np = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_classes = transformed['class_labels']

        # 4. Reconstruct Targets Tensor
        if len(transformed_bboxes) > 0:
            # Stitch class ID and [x, y, w, h] back together
            new_labels = np.column_stack((transformed_classes, transformed_bboxes))
            target_tensor = torch.from_numpy(new_labels).float()
        else:
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)

        # 5. Convert Image to Tensor
        # Convert HWC to CHW, and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Optional: Apply strictly non-geometric transforms (like ColorJitter) here
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, target_tensor
    
import random
from torch.utils.data import Sampler

class InjectClassSampler(Sampler):
    """Custom sampler that shuffles training indices and injects 
    one random validation index containing the target class each epoch"""
    def __init__(self, train_indices, val_target_indices,num_val_images_to_inject=1):
        self.train_indices = train_indices
        self.val_target_indices = val_target_indices
        self.num_val_images_to_inject = num_val_images_to_inject

    def __iter__(self):
        # 1. Shuffle the training dataset indices
        shuffled_train = list(self.train_indices)
        random.shuffle(shuffled_train)

        # 2. Pick ONE random image index from the validation set containing 'triangle'
        if self.val_target_indices:
            injected_val_idx = random.sample(self.val_target_indices, min(self.num_val_images_to_inject, len(self.val_target_indices)))
            # Insert it into the epoch (e.g., appending it at the end, or you can insert randomly)
            epoch_indices = shuffled_train + injected_val_idx
        else:
            # Fallback in case no triangles were found
            epoch_indices = shuffled_train

        return iter(epoch_indices)

    def __len__(self):
        # Length is training set + 1 injected validation image
        return len(self.train_indices) + min(self.num_val_images_to_inject, len(self.val_target_indices))    
        
    

def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO.
    Args:
        batch: List of tuples (img_tensor, target_tensor)
    Returns:
        imgs: Stacked image tensor [B, C, H, W]
        targets: Stacked target tensor [N_targets, 6] -> (batch_idx, cls, x, y, w, h)
    """
    imgs, targets = zip(*batch)
    
    # 1. Stack images (they must be same size! Resize in __getitem__ if needed)
    imgs = torch.stack(imgs, 0)
    
    # 2. Reformat targets to include batch index
    # Input targets is a tuple of tensors: ([N1, 5], [N2, 5], ...)
    new_targets = []
    for i, t in enumerate(targets):
        if t.numel() > 0: # If image has objects
            # Create batch index column filled with 'i'
            batch_idx = torch.full((t.shape[0], 1), i)
            # Cat [batch_idx, cls, x, y, w, h]
            new_targets.append(torch.cat((batch_idx, t), 1))
    
    # Stack all targets into one big [Total_Objects, 6] tensor
    if new_targets:
        targets = torch.cat(new_targets, 0)
    else:
        # Handle batch with 0 objects (rare but possible)
        targets = torch.zeros((0, 6))
        
    return imgs, targets    

if __name__ == "__main__":
    train_img_dir = "datasets/train/images"
    train_labels_dir = "datasets/train/labels"
    dataset = YOLODataset(train_img_dir,train_labels_dir)
    sample_1,target_1 = dataset[0]

    print("end of example")

