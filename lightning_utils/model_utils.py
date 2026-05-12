import lightning as L

import sys
import os

# Adds the parent directory to the search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from loss.loss import DetectionLoss,E2EDetectLoss
from modules.detect import YOLOE,YOLOE11,YOLOEv26,YOLOv8,YOLO11,YOLOv26
import torch
import torch.optim as optim
import lightning as L
from helpers.helper_functions import prepare_model
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.utils import draw_bounding_boxes

import torch
import lightning as L
from torchvision.utils import draw_bounding_boxes
from ultralytics.utils.nms import non_max_suppression

class ImageTrackingCallback(L.Callback):
    """
    Callback class to visualize model prediction results for a fixed set of images in TensorBoard.
    Dynamically supports both standard YOLO architectures (NMS) and YOLOv10/v11/v26 architectures (NMS-Free End2End).
    """
    def __init__(self, val_dataloader, class_names, num_images=3, log_every_n_epochs=3):
        super().__init__()
        self.num_images = num_images
        self.log_every_n_epochs = log_every_n_epochs
        self.class_names = class_names
        
        # 1. Cleaner and more robust way to gather the fixed images
        images_gathered = []
        for batch in val_dataloader:
            # Handle both dictionary and standard tuple batches
            imgs = batch['img'] if isinstance(batch, dict) else batch[0]
            images_gathered.append(imgs)
            
            # Stop gathering once we have enough images
            if sum(x.size(0) for x in images_gathered) >= num_images:
                break
                
        # Concatenate and slice exactly the number of images we need
        self.fixed_images = torch.cat(images_gathered, dim=0)[:num_images]

        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Only run every N epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
            
        tensorboard = trainer.logger.experiment
        
        # 1. Move fixed images to the exact device the model is currently using
        imgs = self.fixed_images.to(pl_module.device)
        
        # 2. Forward pass (Lightning handles eval mode natively)
        with torch.no_grad():
            raw_preds = pl_module(imgs)
        
        # =================================================================
        # 3. DYNAMIC POST-PROCESSING (End2End vs Standard)
        # =================================================================
        is_end2end = getattr(pl_module.model.detection_head, 'end2end', False)
        predictions = []
        
        if is_end2end:
            # --- NMS-FREE ROUTE ---
            # preds[0] already contains [Batch, Max_Det, 6]
            inf_preds = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
            
            for p in inf_preds:
                # Just filter by confidence, no NMS required!
                mask = p[:, 4] > 0.25
                predictions.append(p[mask])
        else:
            # --- STANDARD ROUTE ---
            # Requires heavy NMS filtering
            predictions = non_max_suppression(raw_preds, conf_thres=0.25, iou_thres=0.45)
            
        # =================================================================
        
        # 4. Draw boxes and log to TensorBoard
        for i, (img_tensor, pred) in enumerate(zip(imgs, predictions)):
            
            # torchvision requires image to be uint8 (0-255) for drawing
            img_uint8 = (img_tensor * 255).byte()
            
            if pred is not None and len(pred) > 0:
                # pred format is identical for both routes: [x1, y1, x2, y2, conf, cls]
                boxes = pred[:, :4] 
                
                # Create labels showing "ClassName: Confidence%" securely
                box_labels = []
                for conf, cls_id in zip(pred[:, 4], pred[:, 5]):
                    idx = int(cls_id.item())
                    
                    if self.class_names and idx < len(self.class_names):
                        name = self.class_names[idx]
                    else:
                        name = f"C{idx}" 
                        
                    box_labels.append(f"{name}: {conf.item():.2f}")

                img_with_boxes = draw_bounding_boxes(img_uint8, 
                                                     boxes=boxes, 
                                                     labels=box_labels, 
                                                     width=3)
            else:
                img_with_boxes = img_uint8 # No objects detected
            
            # Send both the raw and predicted images to TensorBoard
            tensorboard.add_image(f"Visual_Debug/Image_{i+1}_raw", img_uint8, trainer.current_epoch)
            tensorboard.add_image(f"Visual_Debug/Image_{i+1}_pred", img_with_boxes, trainer.current_epoch)
        

###Use TestImgTrackCallback to show images during testing in tensorboard.
###Inherited class show it in validation,not in test.
class TestImgTrackCallback(ImageTrackingCallback):

    def __init__(self, test_loader,class_names, num_images=100, log_every_n_epochs=1):
        super().__init__(test_loader,class_names, num_images, log_every_n_epochs)

    def on_test_epoch_end(self,trainer,pl_module):
        return self.on_validation_epoch_end(trainer=trainer,pl_module=pl_module)
    

class YOLOTrainingMixin:
    def forward(self, x):
        """Standard forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        The training loop.
        """
        # Unpack the batch
        # Assuming dataloader returns a dict like Ultralytics:
        # {'img': [B, 3, H, W], 'cls': [N], 'bboxes': [N, 4], 'batch_idx': [N]}
        # OR standard tuple: (images, targets)
        
        if isinstance(batch, dict):
            imgs = batch['img']
            # Ultralytics loss expects the whole 'batch' dict for targets
            targets = batch 
        else:
            # If using standard PyTorch loader (images, targets)
            imgs, targets = batch
        # 0. Maybe need to preprocess targets here to match v8DetectionLoss format
        ###v8DetectionLoss has preprocess method.just use this method
            

        # 1. Forward Pass
        # Returns list of 3 feature maps (because model.training is True)
        preds = self.model(imgs)

        # 2.Postprocess predictions(NMS and else(debug yoloe val_skadi))
        ###As postprocess can use NMS method which I can debugging val_skadi.py
            
        # We must convert the [N, 6] tensor into the dictionary the loss expects.
        # targets shape is [N, 6] -> (batch_idx, cls, x, y, w, h)
        loss_inputs = {
            'batch_idx': targets[:, 0],     # Column 0 is batch index
            'cls':       targets[:, 1],     # Column 1 is class label
            'bboxes':    targets[:, 2:]     # Columns 2-5 are x,y,w,h
        }

        # 3. Calculate Loss
        # v8DetectionLoss(preds, targmap_metricets)
        # Unpack the two returns
        # loss_for_optim: [Box, Cls, DFL] * Batch_Size (Has Gradients)
        # loss_for_log:   [Box, Cls, DFL] (Detached, Normalized)
        loss_for_optim, loss_for_log = self.loss_fn(preds, loss_inputs)

        # Lightning needs a single scalar to call backward() on
        total_loss = loss_for_optim.sum()


        # 4. Logging
        # loss_items usually contains [box_loss, cls_loss, dfl_loss]
        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/box_loss", loss_for_log[0])       # Box component
        self.log("train/cls_loss", loss_for_log[1])       # Class component
        self.log("train/dfl_loss", loss_for_log[2])       # DFL component
        
        return total_loss


    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, prefix="test")

    def _shared_eval_step(self, batch, batch_idx, prefix):
        """Unified logic for validation and test steps."""
        imgs, targets = batch
        
        # 1. Forward Pass
        preds = self.model(imgs)

        # 3. Calculate Loss (Optional, for monitoring)
        loss_inputs = {
            'batch_idx': targets[:, 0],
            'cls':       targets[:, 1],
            'bboxes':    targets[:, 2:]
        }
        
        # We use the SECOND return value (loss_detached)
        # It contains [box_loss, cls_loss, dfl_loss] already normalized
        _, loss_detached = self.loss_fn(preds, loss_inputs)

        # Sum components to get "Total Validation Loss"
        total_loss = loss_detached.sum()

        # 4.Log
        # Use sync_dist=True if you plan to use multiple GPUs later
        self.log(f"{prefix}/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/box_loss", loss_detached[0], sync_dist=True)
        self.log(f"{prefix}/cls_loss", loss_detached[1], sync_dist=True)
        self.log(f"{prefix}/dfl_loss", loss_detached[2], sync_dist=True)

        # 5. Post-Process: Non-Max Suppression
        # Returns list of tensors: [Det1, Det2, ...] where Det is [x1,y1,x2,y2,conf,cls]
        final_preds = non_max_suppression(preds,conf_thres=.2)

        # 6. Format for MeanAveragePrecision Metric
        # We need to convert stacked targets back to per-image list
        target_list = self.process_batch_for_metric(targets, imgs.shape)
        
        # Convert preds list to dictionary format for Metric
        pred_list = []
        for p in final_preds:
            pred_list.append({
                "boxes": p[:, :4],   # xyxy
                "scores": p[:, 4],
                "labels": p[:, 5].long()
            })

        # 7. Update Metric
        self.map_metric.update(pred_list, target_list)

    def on_validation_epoch_end(self):
        # Compute mAP at end of epoch
        metrics = self.map_metric.compute()
        # 2. Log Overall Metrics
        self.log("val/mAP_50", metrics['map_50'], prog_bar=True, sync_dist=True)
        self.log("val/Recall_100", metrics['mar_100'], prog_bar=True, sync_dist=True)
        
        print(f"\n" + "="*32)
        print(f"🏆 OVERALL VALIDATION RESULTS")
        print(f"="*32)
        print(f"mAP@50:        {metrics['map_50'].item():.4f}")
        print(f"mAP@50-95:     {metrics['map'].item():.4f}")
        print(f"Recall@100:    {metrics['mar_100'].item():.4f}")
        
        # 3. Print Class-wise Metrics!
        print(f"\n📊 CLASS-WISE mAP@50")
        print(f"-"*32)
        
        # metrics['classes'] contains the class IDs that were found in the data
        # metrics['map_50_per_class'] contains their corresponding AP50 scores
        class_ids = metrics['classes']
        ap_per_class = metrics['map_per_class']
        
        # If you saved your class names in __init__, you can use them here!
        # e.g., class_names = self.hparams.text_prompt 
        
        for i, cls_id in enumerate(class_ids):
            class_index = int(cls_id.item())
            class_ap = ap_per_class[i].item()
            
            # Optional: Map ID to name if you have self.class_names
            # class_name = self.class_names[class_index]
            # print(f"Class {class_index} ({class_name}): {class_ap50:.4f}")
            
            print(f"Class {class_index}: {class_ap:.4f}")
            
            # You can also log them to TensorBoard/Wandb individually
            self.log(f"val/mAP_50_class_{class_index}", class_ap, sync_dist=True)
            
        print(f"="*32 + "\n")
        self.map_metric.reset()

    def on_test_epoch_end(self):
        # Compute mAP at end of epoch
        metrics = self.map_metric.compute()
        # 2. Log Overall Metrics
        self.log("test/mAP_50", metrics['map_50'], prog_bar=True, sync_dist=True)
        self.log("test/Recall_100", metrics['mar_100'], prog_bar=True, sync_dist=True)
        
        print(f"\n" + "="*32)
        print(f"🏆 OVERALL TEST RESULTS")
        print(f"="*32)
        print(f"mAP@50:        {metrics['map_50'].item():.4f}")
        print(f"mAP@50-95:     {metrics['map'].item():.4f}")
        print(f"Recall@100:    {metrics['mar_100'].item():.4f}")
        
        # 3. Print Class-wise Metrics!
        print(f"\n📊 CLASS-WISE mAP@50")
        print(f"-"*32)
        
        # metrics['classes'] contains the class IDs that were found in the data
        # metrics['map_50_per_class'] contains their corresponding AP50 scores
        class_ids = metrics['classes']
        ap_per_class = metrics['map_per_class']
        
        # If you saved your class names in __init__, you can use them here!
        # e.g., class_names = self.hparams.text_prompt 
        
        for i, cls_id in enumerate(class_ids):
            class_index = int(cls_id.item())
            class_ap = ap_per_class[i].item()
            
            # Optional: Map ID to name if you have self.class_names
            # class_name = self.class_names[class_index]
            # print(f"Class {class_index} ({class_name}): {class_ap50:.4f}")
            
            print(f"Class {class_index}: {class_ap:.4f}")
            
            # You can also log them to TensorBoard/Wandb individually
            self.log(f"test/mAP_50_class_{class_index}", class_ap, sync_dist=True)
            
        print(f"="*32 + "\n")
        self.map_metric.reset()


    def _fix_model_strides_and_anchors(self):
        """Helper to force correct strides and reset broken anchors."""
        device = self.device
        
        # 1. Force the correct stride directly onto the head
        correct_stride = torch.tensor([8.0, 16.0, 32.0], device=device)
        self.model.detection_head.stride = correct_stride
        
        # 2. THE FIX: Force YOLO to regenerate anchors on the next forward pass
        # By setting shape to None, the model ignores its broken cache
        self.model.detection_head.shape = None
            
        # 3. Also make sure the Loss Function has the updated stride
        if hasattr(self, 'loss_fn'):
            self.loss_fn.stride = correct_stride
            self.loss_fn.device = device
            if hasattr(self.loss_fn, 'bbox_loss'):
                self.loss_fn.bbox_loss = self.loss_fn.bbox_loss.to(device)
            if hasattr(self.loss_fn, 'proj'):
                self.loss_fn.proj = self.loss_fn.proj.to(device)    


    def on_fit_start(self):
        """
        Called once before training begins. 
        Moves the loss function's internal tensors to the correct device (GPU).
        Freeze all parameters but classification head if linear_probing is set True.
        """

        #********Parameters freezing part is moved to initialization function*********#
        # if self.hparams.linear_probing:
        #     for name,param in self.model.named_parameters():
        #         if "detection_head.cv3" in name:
        #             param.requires_grad = True
                    
        #         else:
        #             param.requires_grad = False

        self._fix_model_strides_and_anchors()
        
    # You should also add this for testing/validation only runs
    def on_test_start(self):
        self._fix_model_strides_and_anchors()
        
    def on_validation_start(self):
        self._fix_model_strides_and_anchors()   

    def process_batch_for_metric(self, targets, img_shape):
        """
        Unstacks the YOLO target tensor [N, 6] back into a list of dictionaries.
        Also converts normalized xywh -> absolute xyxy for the metric.
        """
        batch_size = img_shape[0]
        h, w = img_shape[2], img_shape[3]
        target_list = []

        for i in range(batch_size):
            # Select targets belonging to this image index
            mask = targets[:, 0] == i
            t = targets[mask]
            
            if len(t) > 0:
                # Extract classes and boxes
                labels = t[:, 1].long()
                boxes_norm = t[:, 2:] # cx, cy, w, h (normalized 0-1)

                # Convert Normalized cxcywh -> Absolute pixel xyxy
                # 1. Scale to pixels
                boxes_scaled = boxes_norm * torch.tensor([w, h, w, h], device=self.device)
                # 2. Convert format
                boxes_xyxy = xywh2xyxy(boxes_scaled)

                target_list.append({
                    "boxes": boxes_xyxy,
                    "labels": labels
                })
            else:
                # Handle images with no objects
                target_list.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=self.device)
                })
        
        return target_list

    def configure_optimizers(self):
        """
        Setup AdamW optimizer and Cosine Annealing scheduler.
        """
        # Filter params to only train those that require grad
        # (This automatically ignores the frozen text model if it was still there)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Optional: Learning Rate Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100, # Adjust based on your max_epochs
            eta_min=self.hparams.lr * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }



class Yoloev8Lightning(YOLOTrainingMixin,L.LightningModule):
    def __init__(self, text_prompt:list[str],lr=1e-3, weight_decay=5e-4,linear_probing:bool = True,
                 weights_path="models/yoloe-v8l-statedict.pt",only_cv3:bool = True):
        super().__init__()
        self.map_metric = MeanAveragePrecision(box_format='xyxy',###The non_max_suppression function returns boxes in XYXY format (xmin, ymin, xmax, ymax).
                                               iou_type="bbox",
                                               class_metrics=True)
        self.save_hyperparameters() # Saves lr, text_prompt etc to hparams
        
        # 1. Initialize your custom YOLOE model
        # text_model_name can be hardcoded or passed as arg
        self.model = YOLOE(text_model_name="mobileclip:s0", class_names=text_prompt)

        ##delete savpe since we don't use visual prompts
        ##delete it before loading weights from checkpoint file
        del self.model.detection_head.savpe

        ###load weights of pretrained yoloe model.
        if weights_path is not None:
            print("preparing model:loading pre trained weigths and freezing backbone (if linear_probing set True)")
            prepare_model(self.model,weights_path,linear_probing=linear_probing,backbone_version='v8',only_cv3=only_cv3)
        

        # 2. Initialize Loss
        # DetectionLoss needs the model instance to access 'stride' and 'nc'
        # It uses these to dynamically build anchors
        
        self.loss_fn = DetectionLoss(self.model) 



class Yoloe11Lightning(YOLOTrainingMixin,L.LightningModule):
    def __init__(self, text_prompt, lr=0.001, weight_decay=0.0005, linear_probing = True, weights_path=None,scale = 'l',only_cv3:bool = True):
        super().__init__()
        self.map_metric = MeanAveragePrecision(box_format='xyxy',###The non_max_suppression function returns boxes in XYXY format (xmin, ymin, xmax, ymax).
                                               iou_type="bbox",
                                               class_metrics=True)
        self.save_hyperparameters() # Saves lr, text_prompt etc to hparams
        
        # 1. Initialize your custom YOLOE model
        # text_model_name can be hardcoded or passed as arg
        self.model = YOLOE11(text_model_name="mobileclip:s0", class_names=text_prompt,scale=scale)

        ##delete savpe since we don't use visual prompts
        ##delete it before loading weights from checkpoint file
        del self.model.detection_head.savpe

        ###load weights of pretrained yoloe model.
        if weights_path is not None:
            print("preparing model:loading pre trained weigths and freezing backbone (if linear_probing set True)")
            prepare_model(self.model,weights_path,linear_probing=linear_probing,backbone_version='11',only_cv3=only_cv3)
        

        # 2. Initialize Loss
        # DetectionLoss needs the model instance to access 'stride' and 'nc'
        # It uses these to dynamically build anchors
        
        self.loss_fn = DetectionLoss(self.model) 


class YoloeV26Lightning(YOLOTrainingMixin,L.LightningModule):
    def __init__(self, text_prompt, lr=0.001, weight_decay=0.0005, linear_probing = True, weights_path=None,scale = 'l',conf_thres = 0.2,only_cv3:bool = True):
        super().__init__()
        self.conf_thres = conf_thres
        self.map_metric = MeanAveragePrecision(box_format='xyxy',
                                               iou_type="bbox",
                                               class_metrics=True)
        self.save_hyperparameters() # Saves lr, text_prompt etc to hparams
        
        # 1. Initialize your custom YOLOE model
        # text_model_name can be hardcoded or passed as arg
        self.model = YOLOEv26( class_names=text_prompt,scale=scale)

        ##delete savpe since we don't use visual prompts
        ##delete it before loading weights from checkpoint file
        del self.model.detection_head.savpe

        ###load weights of pretrained yoloe model.
        if weights_path is not None:
            print("preparing model:loading pre trained weigths and freezing backbone (if linear_probing set True,and train all detection_head)")
            prepare_model(self.model,weights_path,linear_probing=linear_probing,backbone_version='v26',only_cv3=only_cv3)
        

        # 2. Initialize Loss
        # DetectionLoss needs the model instance to access 'stride' and 'nc'
        # It uses these to dynamically build anchors
        
        self.loss_fn = E2EDetectLoss(self.model) 

    ###Overwrite _shared_eval_step to match with END2END
    def _shared_eval_step(self, batch, batch_idx, prefix):
        """Unified logic for validation and test steps."""
        imgs, targets = batch
        
        # 1. Forward Pass
        preds = self.model(imgs)

        # 3. Calculate Loss (Optional, for monitoring)
        loss_inputs = {
            'batch_idx': targets[:, 0],
            'cls':       targets[:, 1],
            'bboxes':    targets[:, 2:]
        }
        
        # We use the SECOND return value (loss_detached)
        # It contains [box_loss, clfl_s_loss, dloss] already normalized
        _, loss_detached = self.loss_fn(preds, loss_inputs)

        # Sum components to get "Total Validation Loss"
        total_loss = loss_detached.sum()

        # 4.Log
        # Use sync_dist=True if you plan to use multiple GPUs later
        self.log(f"{prefix}/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/box_loss", loss_detached[0], sync_dist=True)
        self.log(f"{prefix}/cls_loss", loss_detached[1], sync_dist=True)
        self.log(f"{prefix}/dfl_loss", loss_detached[2], sync_dist=True)

        # ==========================================
        # 5. POST-PROCESS: THE NMS-FREE MAGIC
        # ==========================================
        # preds[0] contains the pre-filtered one-to-one predictions
        # Shape is [Batch, Max_Det, 6] -> Max_Det is usually 300
        inf_preds = preds[0] if isinstance(preds, tuple) else preds
        
        pred_list = []
        for p in inf_preds:
            # p is [300, 6] -> [x1, y1, x2, y2, conf, cls]
            
            # We simply filter out the low-confidence boxes with a mask
            # No Non-Maximum Suppression needed!
            conf_mask = p[:, 4] > self.conf_thres  # conf_thres = 0.2
            valid_boxes = p[conf_mask]
            
            pred_list.append({
                "boxes": valid_boxes[:, :4],   # xyxy
                "scores": valid_boxes[:, 4],
                "labels": valid_boxes[:, 5].long()
            })

        # 6. Format for MeanAveragePrecision Metric
        target_list = self.process_batch_for_metric(targets, imgs.shape)

        # 7. Update Metric
        self.map_metric.update(pred_list, target_list)

    

    def _fix_model_strides_and_anchors(self):
        """Helper to force correct strides and reset broken anchors."""
        device = self.device
        
        # 1. Force the correct stride directly onto the head
        correct_stride = torch.tensor([8.0, 16.0, 32.0], device=device)
        self.model.detection_head.stride = correct_stride
        
        # 2. THE FIX: Force YOLO to regenerate anchors on the next forward pass
        # By setting shape to None, the model ignores its broken cache
        self.model.detection_head.shape = None
            
        # 3. Update the Loss Function(s)
        if hasattr(self, 'loss_fn'):
            # Create a list of loss objects to update. 
            # If it's E2E, we update both branches. If standard v8, just the main one.
            if hasattr(self.loss_fn, 'one2many'):
                losses_to_sync = [self.loss_fn.one2many, self.loss_fn.one2one]
            else:
                losses_to_sync = [self.loss_fn]
                
            # Loop through and explicitly force everything to the correct GPU
            for loss_obj in losses_to_sync:
                loss_obj.stride = correct_stride
                loss_obj.device = device
                
                if hasattr(loss_obj, 'bbox_loss'):
                    loss_obj.bbox_loss = loss_obj.bbox_loss.to(device)
                if hasattr(loss_obj, 'proj'):
                    loss_obj.proj = loss_obj.proj.to(device)


 

class Yolov8Lighning(YOLOTrainingMixin,L.LightningModule):
    def __init__(self,nc,scale, weights_path, lr=0.001, weight_decay=0.0005, linear_probing = True,only_cv3:bool = True):
        """
        params
            nc:number of classes
            scale:scale of yolo model(for example "l" means large model)
            only_cv3: if linear_probing set True then if only_cv3 is set As True then only classification head will be trained
        """
        super().__init__()
        self.map_metric = MeanAveragePrecision(box_format='xyxy',###The non_max_suppression function returns boxes in XYXY format (xmin, ymin, xmax, ymax).
                                               iou_type="bbox",
                                               class_metrics=True)
        self.save_hyperparameters() # Saves lr, text_prompt etc to hparams
        
        # 1. Initialize your custom YOLOE model
        # text_model_name can be hardcoded or passed as arg
        self.model = YOLOv8(nc = nc,scale = scale)

        ###load weights of pretrained yoloe model.
        if weights_path is not None:
            print("preparing model:loading pre trained weigths and freezing backbone (if linear_probing set True)")
            ###Set Strict to False because channel numbers of classification head must be different then checkpoint file,which used 80 coco classes
            prepare_model(self.model,weights_path,linear_probing=linear_probing,backbone_version='v8',only_cv3=only_cv3,strict=False)
        

        # 2. Initialize Loss
        # DetectionLoss needs the model instance to access 'stride' and 'nc'
        # It uses these to dynamically build anchors
        
        self.loss_fn = DetectionLoss(self.model) 

class Yolo11Lightning(YOLOTrainingMixin,L.LightningModule):       
    def __init__(self,nc,scale, weights_path, lr=0.001, weight_decay=0.0005, linear_probing = True,only_cv3:bool = True):
        """
        params
            nc:number of classes
            scale:scale of yolo model(for example "l" means large model)
            only_cv3: if linear_probing set True then if only_cv3 is set As True then only classification head will be trained
        """
        super().__init__()
        
        self.map_metric = MeanAveragePrecision(box_format='xyxy',###The non_max_suppression function returns boxes in XYXY format (xmin, ymin, xmax, ymax).
                                               iou_type="bbox",
                                               class_metrics=True)

        self.save_hyperparameters() # Saves lr, text_prompt etc to hparams
        
        # 1. Initialize your custom YOLO model
        self.model = YOLO11(nc = nc,scale=scale)

        ###load weights of pretrained yoloe model.
        if weights_path is not None:
            print("preparing model:loading pre trained weigths and freezing backbone (if linear_probing set True)")
            ###Set Strict to False because channel numbers of classification head must be different then checkpoint file,which used 80 coco classes
            prepare_model(self.model,weights_path,linear_probing=linear_probing,backbone_version='11',only_cv3=only_cv3,strict=False)
        

        # 2. Initialize Loss
        # DetectionLoss needs the model instance to access 'stride' and 'nc'
        # It uses these to dynamically build anchors
        
        self.loss_fn = DetectionLoss(self.model) 


class Yolo26Lightning(YOLOTrainingMixin,L.LightningModule):
    def __init__(self,nc,scale, weights_path, lr=0.001, weight_decay=0.0005, linear_probing = True,only_cv3:bool = True,conf_thres = 0.2):
        """
        params
            nc:number of classes
            scale:scale of yolo model(for example "l" means large model)
            only_cv3: if linear_probing set True then if only_cv3 is set As True then only classification head will be trained
            conf_thres:confidence threshold to take only predicitions which have bigger confidence during inference
        """
        super().__init__()


        self.conf_thres = conf_thres
        self.map_metric = MeanAveragePrecision(box_format='xyxy',
                                               iou_type="bbox",
                                               class_metrics=True)
        self.save_hyperparameters() # Saves lr, text_prompt etc to hparams
        
        # 1. Initialize your custom YOLOE model
        # text_model_name can be hardcoded or passed as arg
        self.model = YOLOv26(nc = nc,scale = scale)


        ###load weights of pretrained yoloe model.
        if weights_path is not None:
            print("preparing model:loading pre trained weigths and freezing backbone (if linear_probing set True,and train all detection_head)")
            ###Set Strict to False because channel numbers of classification head must be different then checkpoint file,which used 80 coco classes
            prepare_model(self.model,weights_path,linear_probing=linear_probing,backbone_version='v26',only_cv3=only_cv3,strict=False)
        

        # 2. Initialize Loss
        # DetectionLoss needs the model instance to access 'stride' and 'nc'
        # It uses these to dynamically build anchors
        
        self.loss_fn = E2EDetectLoss(self.model) 


    ###We need to overwrite these 2 methods,but then understand why Mixin methods don't work for this case,precisely device mismtach problem occurs
       ###Overwrite _shared_eval_step to match with END2END
    def _shared_eval_step(self, batch, batch_idx, prefix):
        """Unified logic for validation and test steps."""
        imgs, targets = batch
        
        # 1. Forward Pass
        preds = self.model(imgs)

        # 3. Calculate Loss (Optional, for monitoring)
        loss_inputs = {
            'batch_idx': targets[:, 0],
            'cls':       targets[:, 1],
            'bboxes':    targets[:, 2:]
        }
        
        # We use the SECOND return value (loss_detached)
        # It contains [box_loss, clfl_s_loss, dloss] already normalized
        _, loss_detached = self.loss_fn(preds, loss_inputs)

        # Sum components to get "Total Validation Loss"
        total_loss = loss_detached.sum()

        # 4.Log
        # Use sync_dist=True if you plan to use multiple GPUs later
        self.log(f"{prefix}/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/box_loss", loss_detached[0], sync_dist=True)
        self.log(f"{prefix}/cls_loss", loss_detached[1], sync_dist=True)
        self.log(f"{prefix}/dfl_loss", loss_detached[2], sync_dist=True)

        # ==========================================
        # 5. POST-PROCESS: THE NMS-FREE MAGIC
        # ==========================================
        # preds[0] contains the pre-filtered one-to-one predictions
        # Shape is [Batch, Max_Det, 6] -> Max_Det is usually 300
        inf_preds = preds[0] if isinstance(preds, tuple) else preds
        
        pred_list = []
        for p in inf_preds:
            # p is [300, 6] -> [x1, y1, x2, y2, conf, cls]
            
            # We simply filter out the low-confidence boxes with a mask
            # No Non-Maximum Suppression needed!
            conf_mask = p[:, 4] > self.conf_thres  # conf_thres = 0.2
            valid_boxes = p[conf_mask]
            
            pred_list.append({
                "boxes": valid_boxes[:, :4],   # xyxy
                "scores": valid_boxes[:, 4],
                "labels": valid_boxes[:, 5].long()
            })

        # 6. Format for MeanAveragePrecision Metric
        target_list = self.process_batch_for_metric(targets, imgs.shape)

        # 7. Update Metric
        self.map_metric.update(pred_list, target_list)

    

    def _fix_model_strides_and_anchors(self):
        """Helper to force correct strides and reset broken anchors."""
        device = self.device
        
        # 1. Force the correct stride directly onto the head
        correct_stride = torch.tensor([8.0, 16.0, 32.0], device=device)
        self.model.detection_head.stride = correct_stride
        
        # 2. THE FIX: Force YOLO to regenerate anchors on the next forward pass
        # By setting shape to None, the model ignores its broken cache
        self.model.detection_head.shape = None
            
        # 3. Update the Loss Function(s)
        if hasattr(self, 'loss_fn'):
            # Create a list of loss objects to update. 
            # If it's E2E, we update both branches. If standard v8, just the main one.
            if hasattr(self.loss_fn, 'one2many'):
                losses_to_sync = [self.loss_fn.one2many, self.loss_fn.one2one]
            else:
                losses_to_sync = [self.loss_fn]
                
            # Loop through and explicitly force everything to the correct GPU
            for loss_obj in losses_to_sync:
                loss_obj.stride = correct_stride
                loss_obj.device = device
                
                if hasattr(loss_obj, 'bbox_loss'):
                    loss_obj.bbox_loss = loss_obj.bbox_loss.to(device)
                if hasattr(loss_obj, 'proj'):
                    loss_obj.proj = loss_obj.proj.to(device)

    
