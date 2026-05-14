from dataset_constructor import YOLODataset,yolo_collate_fn
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from torchvision import transforms
from torch.utils.data import random_split
import torch
from pathlib import Path
from loguru import logger
from dataset_constructor import InjectClassSampler
from helpers.data_helpers import get_yolo_target_indices
import lightning as pl

class SyntheticYoloDataModule(pl.LightningDataModule):
    """DataModule that combines training and validation datasets,
      and injects one random validation image containing the target class into each training epoch.
      parameters:
      - train_dataset: PyTorch Dataset for training data
      - val_dataset: PyTorch Dataset for validation data
      - test_dataset: PyTorch Dataset for test data
      - val_label_dir: Directory containing YOLO txt label files for the validation set
      - num_val_images_to_inject: How many images randomly sample from validation dataset which includes objects of class target_class_id
      - target_class_id: The class ID to look for in the validation labels (default '2' for triangle)
      - batch_size: Batch size for DataLoaders
      """
    def __init__(self,train_dataset:None|Dataset,
                        val_dataset:None|Dataset,
                        test_dataset:None|Dataset, 
                        val_label_dir: str|Path,
                        target_class_id: str = '2',
                        num_val_images_to_inject: int = 1,
                        num_workers: int = 4,
                        batch_size=16):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.val_label_dir = val_label_dir
        self.target_class_id = target_class_id
        self.batch_size = batch_size
        self.num_val_images_to_inject = num_val_images_to_inject
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 1. Combine Datasets
        self.combined_dataset = ConcatDataset([self.train_dataset, self.val_dataset]) if self.val_dataset is not None else self.train_dataset
        if self.val_dataset is None:
            logger.warning("Validation dataset is None. Cannot inject validation images into training epochs.")
        
        
        # 2. Setup Indices
        train_len = len(self.train_dataset)
        if train_len == 0:
            logger.warning("Training dataset is empty. Cannot create training dataloader.")
            self.train_indices = []
            return
        self.train_indices = list(range(train_len))
        
        # 3. Find validation indices with the target class
        # We pass 'train_len' as an offset because the val data comes AFTER 
        # the train data in the ConcatDataset.
        self.val_target_indices = get_yolo_target_indices(
            val_label_dir=self.val_label_dir, 
            target_class_id=self.target_class_id, 
            offset=train_len
        )
        
        logger.info(f"Found {len(self.val_target_indices)} validation images with the {self.target_class_id} class.")

    def train_dataloader(self):
        # 4. Use the custom sampler
        if not self.train_dataset:
            logger.warning("Training dataset is None. Cannot create training dataloader.")
            return None
        sampler = InjectClassSampler(self.train_indices, self.val_target_indices,num_val_images_to_inject=self.num_val_images_to_inject) # You can adjust how many validation images to inject per epoch
        
        return DataLoader(
            self.combined_dataset, 
            sampler=sampler, 
            batch_size=self.batch_size,
            # Note: shuffle must be False when using a custom Sampler, 
            # because the Sampler handles the shuffling internally.
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=yolo_collate_fn
        )

    def val_dataloader(self):
        if not self.val_dataset:
            logger.warning("Validation dataset is None. Cannot create validation dataloader.")
            return None
        # Validation remains completely normal
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=yolo_collate_fn)
    
    def test_dataloader(self):
        if not self.test_dataset:
            logger.warning("Test dataset is None. Cannot create test dataloader.")
            return None
        # Test remains completely normal
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=yolo_collate_fn)
  

def create_dataloader(dataset_dir:str,
                      batch_size:int = 32,
                      transform:transforms.Compose = None,
                      num_workers:int = 4,
                      split:bool = True,
                      split_size = (0.8,0.1),
                      return_lightning_datamodule:bool = False,
                      target_class_id:str = '2',
                      num_val_images_to_inject:int = 1)->tuple[DataLoader,DataLoader,DataLoader]|SyntheticYoloDataModule:
    """
    If split is set False(i.e. dataset is already splited)
    Then dataset directory hieararchy must be:
    dataset_dir
                -train
                    -images
                    -labels
                
                -test
                    -images
                    -labels
    else 
    dataset_dir
                -images
                -labels

    parameters:
    - dataset_dir: Root directory of the dataset
    - batch_size: Batch size for DataLoaders
    - transform: Optional torchvision transforms to apply to images
    - num_workers: Number of subprocesses for data loading
    - split: Whether to split the dataset into train/val/test or assume it's already split
    - split_size: Tuple indicating the proportions for train/val/test splits (only used if split=True)
    - return_lightning_datamodule: Warning:SET THIS TRUE ONLY IN train_forgetting.py. If True, returns a PyTorch Lightning DataModule instead of individual DataLoaders
    - target_class_id: The class ID to look for in the validation labels when using the Lightning DataModule (default '2' for triangle).target_class_id is used to find validation images containing the target class, which are then injected into the training epochs when using the Lightning DataModule.
                                        
    """


    if not split:
        if Path(dataset_dir + "/train").is_dir():
            train_image_dir = dataset_dir + "/train/images"
            train_labels_dir = dataset_dir + "/train/labels"
            train_dataset = YOLODataset(img_dir=train_image_dir,label_dir=train_labels_dir,transform=transform)
        else:
            train_dataset = None
            logger.warning("train folder don't exist")

        if Path(dataset_dir + "/test").is_dir():
            test_image_dir = dataset_dir + "/test/images"
            test_labels_dir = dataset_dir + "/test/labels"
            test_dataset = YOLODataset(img_dir=test_image_dir,label_dir=test_labels_dir,transform=transform)
        else:
            test_dataset = None
            logger.warning("test folder don't exist")

        if Path(dataset_dir + "/valid").is_dir():
            valid_image_dir = dataset_dir + "/valid/images"
            valid_labels_dir = dataset_dir + "/valid/labels"
            val_dataset = YOLODataset(img_dir=valid_image_dir,label_dir=valid_labels_dir,transform=transform)
        else:
            valid_labels_dir = None
            val_dataset = None
            logger.warning("valid folder don't exist")
    else:
        image_dir = dataset_dir + "/images"
        labels_dir = dataset_dir + "/labels"
        dataset = YOLODataset(img_dir=image_dir,label_dir=labels_dir,transform=transform)

        train_size = int( split_size[0]* len(dataset))
        val_size = int( split_size[1]* len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset,val_dataset,test_dataset = random_split(dataset, [train_size,val_size ,test_size],generator=torch.Generator().manual_seed(42))

    if return_lightning_datamodule:
        datamodule = SyntheticYoloDataModule(train_dataset=train_dataset,
                                            val_dataset=val_dataset,
                                            test_dataset=test_dataset,
                                            val_label_dir=valid_labels_dir,
                                            target_class_id=target_class_id,
                                            num_val_images_to_inject=num_val_images_to_inject,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
        return datamodule

    try:
        train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=yolo_collate_fn)
    except:
        train_loader = None
        logger.info("train loader is None")    
        
    try:
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=yolo_collate_fn)
    except:
        val_loader = None
        logger.info("validation loader is None")    

    try:
        test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=yolo_collate_fn)
    except:
        test_loader = None
        logger.info("test loader is None") 
               
    return train_loader,val_loader,test_loader


###Lightning module of creating dataloader for training and evaluation

if __name__ == "__main__":
    datamodule= create_dataloader(dataset_dir="datasets/generated_dataset",
                      batch_size=32,split=False,return_lightning_datamodule=True,target_class_id='2',num_val_images_to_inject=1,num_workers=4)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    