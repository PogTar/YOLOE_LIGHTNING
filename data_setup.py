from dataset_constructor import YOLODataset,yolo_collate_fn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.utils.data import random_split
import torch
from pathlib import Path
from loguru import logger

def create_dataloader(dataset_dir:str,
                      batch_size:int = 32,
                      transform:transforms.Compose = None,
                      num_workers:int = 4,
                      split:bool = True,
                      split_size = (0.8,0.1))->tuple[DataLoader,DataLoader]:
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

    example of using this function to create dataloader only for evaluation:
                                        
    """


    if not split:
        if Path(dataset_dir + "/train").is_dir():
            train_image_dir = dataset_dir + "/train/images"
            train_labels_dir = dataset_dir + "/train/labels"
            train_dataset = YOLODataset(img_dir=train_image_dir,label_dir=train_labels_dir,transform=transform)
        else:
            logger.warning("train folder don't exist")

        if Path(dataset_dir + "/test").is_dir():
            test_image_dir = dataset_dir + "/test/images"
            test_labels_dir = dataset_dir + "/test/labels"
            test_dataset = YOLODataset(img_dir=test_image_dir,label_dir=test_labels_dir,transform=transform)
        else:
            logger.warning("test folder don't exist")

        if Path(dataset_dir + "/valid").is_dir():
            valid_image_dir = dataset_dir + "/valid/images"
            valid_labels_dir = dataset_dir + "/valid/labels"
            val_dataset = YOLODataset(img_dir=valid_image_dir,label_dir=valid_labels_dir,transform=transform)
        else:
            logger.warning("valid folder don't exist")
    else:
        image_dir = dataset_dir + "/images"
        labels_dir = dataset_dir + "/labels"
        dataset = YOLODataset(img_dir=image_dir,label_dir=labels_dir,transform=transform)

        train_size = int( split_size[0]* len(dataset))
        val_size = int( split_size[1]* len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset,val_dataset,test_dataset = random_split(dataset, [train_size,val_size ,test_size],generator=torch.Generator().manual_seed(42))

    
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
    
if __name__ == "__main__":
    train_loader,val_loader,test_loader = create_dataloader(dataset_dir="datasets/generated_dataset",
                      batch_size=32,split=False)
    next(iter(train_loader))
    