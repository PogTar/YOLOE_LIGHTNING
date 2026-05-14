import torch
import cv2
from pathlib import Path
from torchvision.ops import box_iou
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import  scale_boxes, xywh2xyxy
from ultralytics.utils.nms import non_max_suppression
from lightning_utils.model_utils import YoloeLightning


# 1. Define your paths
IMAGE_DIR = Path("datasets/skadi_in_yolo_format/images") 
GT_LABEL_DIR = Path("datasets/skadi_in_yolo_format/labels")
OUTPUT_LABEL_DIR = Path("datasets/skadi_in_yolo_format/pseudolabels")
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.5

def load_yolo_txt(txt_path):
    """Reads a YOLO format txt file and returns a tensor of [class, x_center, y_center, w, h]"""
    if not txt_path.exists():
        return torch.empty((0, 5))
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        data = [float(x) for x in line.strip().split()]
        if len(data) == 5:
            boxes.append(data)
    return torch.tensor(boxes)




def generate_pseudolabels(lightning_model, device="cuda"):
    lightning_model.eval()
    lightning_model.to(device)
    
    image_paths = list(IMAGE_DIR.glob("*.jpg")) # Add *.png if needed
    print(f"🔍 Found {len(image_paths)} images. Generating pseudolabels...")
    
    # Initialize the LetterBox transform once
    letterbox = LetterBox(new_shape=(640, 640))
    
    for img_path in image_paths:
        # 1. Read the matching Ground Truth file
        gt_txt_path = GT_LABEL_DIR / f"{img_path.stem}.txt"
        gt_data = load_yolo_txt(gt_txt_path).to(device) # [N, 5] (cls, xc, yc, w, h)
        
        # 2. Read and format the image for the model
        img_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]  # Save original dimensions!
        
        # REFACTOR: Apply Letterboxing instead of cv2.resize
        img_letterboxed = letterbox(image=img_bgr)
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        # 3. Get Model Predictions
        with torch.no_grad():
            raw_preds = lightning_model(img_tensor)
            # Filter low confidence boxes
            preds = non_max_suppression(raw_preds, conf_thres=0.20)[0] 
        
        final_boxes_to_write = []
        
        # Add all Ground Truth boxes to our final list first (these are perfectly preserved!)
        for i in range(len(gt_data)):
            final_boxes_to_write.append(gt_data[i].cpu().tolist())
            
        # 4. Compare Predictions to Ground Truth
        if preds is not None and len(preds) > 0 and len(gt_data) > 0:
            
            # REFACTOR: Scale predictions from 640x640 padded back to original image size
            # shape[2:] is (640, 640), orig_shape is (orig_h, orig_w)
            pred_boxes_xyxy = scale_boxes(img_tensor.shape[2:], preds[:, :4], (orig_h, orig_w))
            
            # REFACTOR: Convert GT from normalized to absolute original pixels
            gt_boxes_xyxy = xywh2xyxy(gt_data[:, 1:5])
            gt_boxes_xyxy[:, [0, 2]] *= orig_w  # Scale X coordinates to orig_w
            gt_boxes_xyxy[:, [1, 3]] *= orig_h  # Scale Y coordinates to orig_h
            
            # Calculate IoU matrix on the true original pixel scale
            iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy) # Shape: [num_preds, num_gts]
            max_ious, _ = iou_matrix.max(dim=1)
            
            for i, max_iou in enumerate(max_ious):
                if max_iou < IOU_THRESHOLD:
                    # The model found a new object! 
                    x1, y1, x2, y2 = pred_boxes_xyxy[i].tolist()
                    cls_id = int(preds[i, 5].item())
                    
                    # Convert absolute original pixels back to normalized YOLO format
                    xc = ((x1 + x2) / 2) / orig_w
                    yc = ((y1 + y2) / 2) / orig_h
                    w = (x2 - x1) / orig_w
                    h = (y2 - y1) / orig_h
                    
                    final_boxes_to_write.append([cls_id, xc, yc, w, h])
                    
        elif preds is not None and len(preds) > 0 and len(gt_data) == 0:
            # If there was NO ground truth file, keep all predictions
            
            # Must still scale the predictions back to the original image dimensions!
            pred_boxes_xyxy = scale_boxes(img_tensor.shape[2:], preds[:, :4], (orig_h, orig_w))
            
            for i in range(len(preds)):
                x1, y1, x2, y2 = pred_boxes_xyxy[i].tolist()
                cls_id = int(preds[i, 5].item())
                
                # Normalize relative to the true image dimensions
                xc, yc = ((x1 + x2) / 2) / orig_w, ((y1 + y2) / 2) / orig_h
                w, h = (x2 - x1) / orig_w, (y2 - y1) / orig_h
                final_boxes_to_write.append([cls_id, xc, yc, w, h])

        # 5. Write the final Pseudolabel .txt file
        out_txt_path = OUTPUT_LABEL_DIR / f"{img_path.stem}.txt"
        with open(out_txt_path, 'w') as f:
            for box in final_boxes_to_write:
                cls_id = int(box[0])
                xc, yc, w, h = box[1], box[2], box[3], box[4]
                # Write in standard YOLO format: class xc yc w h
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
    print(f"✅ Finished! Pseudolabels saved to: {OUTPUT_LABEL_DIR}")

with open('tools/ram_tag_list.txt','r') as f:
    class_names = [x.strip() for x in f.readlines()]
# from lightning_utils import YoloeLightning
model = YoloeLightning(text_prompt=class_names)
generate_pseudolabels(model)