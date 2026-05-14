import sys
import os

# Adds the parent directory to the search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from data_setup import create_dataloader
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from model_utils import ImageTrackingCallback,Yoloev8Lightning
import yaml
import lightning as L
import torch

dataset_dir = "datasets/synthetic_dataset_for_eval"
dataset_module = create_dataloader(dataset_dir=dataset_dir,
                                    split=False,
                                    return_lightning_datamodule=True,
                                    target_class_id='2',
                                    num_val_images_to_inject=1,
                                    num_workers=0,
                                    batch_size=8)
    
###setup TensorBoard logger
logger = TensorBoardLogger(save_dir="tensorboard_logs",name="yoloev8_training_forgetting_experiment_with_injected_val_triangle_images")

# Open the file and load its content
with open('datasets/synthetic_dataset_for_eval/dataset.yaml', 'r') as file:
    try:
        data = yaml.safe_load(file)
        class_names = list(data["names"].values())
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")

model = Yoloev8Lightning(text_prompt=class_names,lr=3*1e-5,weights_path="models/yoloe-v8l-statedict.pt")         

image_visualization_callback = ImageTrackingCallback(val_dataloader=dataset_module.val_dataloader(),
                                                     num_images=10,
                                                     log_every_n_epochs=3,
                                                     class_names=class_names)

# 1. Define the callback
early_stop_callback = EarlyStopping(
    monitor="val/mAP_50",  # Metric to monitor
    min_delta=0.00,      # Minimum change to qualify as an improvement
    patience=10,          # Number of epochs to wait for improvement
    verbose=False,
    mode="max"           # "min" for loss, "max" for accuracy
)

accelerator = "cpu"
trainer = L.Trainer(callbacks=[early_stop_callback,image_visualization_callback],
                    accelerator=accelerator,
                    max_epochs=15,
                    logger=logger,
                    enable_checkpointing=False)

trainer.fit(model=model,datamodule=dataset_module)