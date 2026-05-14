from lightning_utils import YoloeLightning 
from pathlib import Path
from data_setup import create_dataloader
from lightning_utils import TestImgTrackCallback
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L


    
# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models_checkpoints")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
                )



dataset_dir = "datasets/skadi_in_yolo_format"
test_loader,_ ,_ = create_dataloader(dataset_dir=dataset_dir,split_size=(0.1,0))

with open('tools/ram_tag_list.txt','r') as f:
    class_names = [x.strip() for x in f.readlines()]

lighning_model = YoloeLightning(text_prompt=class_names)

###fuse text_features with parameters  
# lighning_model.model.fuse()

#Setup tensorboard logger
logger = TensorBoardLogger(save_dir="tensorboard_logs",name="prompt_free_evaluation")

###Setup callbacks
image_visualization_callback = TestImgTrackCallback(test_loader=test_loader,
                                                    class_names=class_names,
                                                     num_images=100,
                                                     log_every_n_epochs=1)


trainer = L.Trainer(callbacks=[image_visualization_callback],
                    max_epochs=1,
                    logger=logger,
                    enable_checkpointing=False)


##Run evaluation on a scpecific test set after training
trainer.test(lighning_model,dataloaders=test_loader)
