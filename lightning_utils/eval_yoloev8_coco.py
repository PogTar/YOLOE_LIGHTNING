# import fiftyone as fo
# import fiftyone.zoo as foz

# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#     label_types=["detections"],
#     max_samples=10000,
#     shuffle=True,
# )
# export_dir = "/root/repos/yoloe_lightning/datasets/coco-2017"
# # 3. Export directly to YOLO format
# dataset.export(
#     export_dir=export_dir,
#     dataset_type=fo.types.YOLOv5Dataset,  # This tells FiftyOne to make YOLO .txt files
#     label_field="ground_truth",           # The field containing your labels (usually "ground_truth")
#     split="val",                          # Optional: organizes it into a 'val' folder
# )
import sys
import os
import yaml

# Adds the parent directory to the search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from data_setup import create_dataloader
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from model_utils import ImageTrackingCallback,Yoloev8Lightning
import lightning as L

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir

dataset_dir = "datasets/coco-2017"
_,_, test_loader = create_dataloader(dataset_dir=dataset_dir,split = False)###we already have splited only validation dataset,so we need to set train and validation to 0%.
    
###setup TensorBoard logger
logger = TensorBoardLogger(save_dir="tensorboard_logs",name="yoloe_coco_evaluation_after_fine_tuning")

# Open the file and load its content
with open('datasets/coco-2017/dataset.yaml', 'r') as file:
    try:
        data = yaml.safe_load(file)
        class_names = list(data["names"].values())
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")

model = Yoloev8Lightning(text_prompt=class_names,lr=3*1e-3,weights_path="/root/repos/yoloe_lightning/models_checkpoints/yoloev8/yoloe-linear-probing-skadi-ctzn-objects-lr1e-4-10-0.50-0.47.ckpt")   

image_visualization_callback = ImageTrackingCallback(val_dataloader=test_loader,
                                                     num_images=10,
                                                     log_every_n_epochs=3,
                                                     class_names=class_names)

trainer = L.Trainer(callbacks=[image_visualization_callback],
                    max_epochs=500,
                    logger=logger,
                    enable_checkpointing=False)

trainer.test(model,dataloaders=test_loader)


