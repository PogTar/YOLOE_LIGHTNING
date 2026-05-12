import sys
import os

# Adds the parent directory to the search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from data_setup import create_dataloader
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from model_utils import Yolo11Lightning,ImageTrackingCallback
import lightning as L



# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models_checkpoints/yolo11_withEweights")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
                )



dataset_dir = "datasets/skadi_in_yolo_format"
train_loader,val_loader, test_loader = create_dataloader(dataset_dir=dataset_dir)
    
###setup TensorBoard logger
logger = TensorBoardLogger(save_dir="tensorboard_logs",name="yolo11_withEweights_linearprobing_skadi")


class_names =  ['car', 'truck', 'van_minivan']
# (I mean if in labels triangle :0,rectangle:1,circle:2 then in class_names this map must remain same).

model = Yolo11Lightning(nc = len(class_names),scale = "l",lr = 3*1e-4,weights_path="models/yoloe-11l-seg.pt",only_cv3=False)

# 1. Define the callback
early_stop_callback = EarlyStopping(
    monitor="val/mAP_50",  # Metric to monitor
    min_delta=0.00,      # Minimum change to qualify as an improvement
    patience=10,          # Number of epochs to wait for improvement
    verbose=False,
    mode="max"           # "min" for loss, "max" for accuracy
)

image_visualization_callback = ImageTrackingCallback(val_dataloader=val_loader,
                                                     num_images=30,
                                                     log_every_n_epochs=3,
                                                     class_names=class_names)

checkpoint_callback = ModelCheckpoint(dirpath=MODEL_PATH,
                                      monitor='val/mAP_50',
                                      filename='yolo11-withEweights-linear-probing-skadi-ctzn-objects-lr1e-4-{epoch}-{val/mAP_50:.2f}-{val/Recall_100:.2f}',
                                      mode='max',
                                      auto_insert_metric_name=False # Prevents it from creating folders for slash-names                                                  
)

trainer = L.Trainer(callbacks=[early_stop_callback,image_visualization_callback,checkpoint_callback],
                    max_epochs=500,
                    logger=logger,
                    enable_checkpointing=False,
                    log_every_n_steps=1,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm"
                    )

##Run training and validation during training[]
trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
##Run evaluation on a scpecific test set after training
trainer.test(model,dataloaders=test_loader)

# Save the model state dict

# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model.model.state_dict(), # only saving the state_dict() only saves the learned parameters
#            f=MODEL_SAVE_PATH)
