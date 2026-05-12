from ultralytics.nn.modules.head import YOLOEDetect as YOLOEDetectInit
from ultralytics.nn.modules.head import Detect as DetectInit
import torch
from torch import nn
from torchvision import transforms
from modules.backbone import Yolov8Backbone,Yolo11Backbone,Yolov26Backbone
from modules.head import YOLOv8Head,YOLO11Head,YOLOv26Head
from pathlib import Path
from PIL import Image
from ultralytics.nn.text_model import build_text_model

class Detectv8(DetectInit):
    legacy = True

class Detect11(DetectInit):
    legacy = False

class Detectv26(DetectInit):
    legacy = False

class YOLOEv8Detect(YOLOEDetectInit):
    legacy = True

class YOLOE11Detect(YOLOEDetectInit):
    legacy = False

class YOLOEv26Detect(YOLOEDetectInit):
    legacy = False    

class YOLOv8(nn.Module):
    def __init__(self,nc,scale = "l",*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        scale_map = {
            'n': [0.33, 0.25, 1024],
            's': [0.33, 0.5, 1024],
            'm': [0.67, 0.75, 768],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.25, 512]
        }
        
        # Extract depth (d), width (w), and max_channels (max_c)
        _, w, max_c = scale_map[scale]

        ch = (int(min(256 * w, max_c)),int(min(512 * w, max_c)),int(min(1024 * w, max_c)))

        self.args = Arguments()###Arguments class set hypermeremeters like lambdas for loss function components,dropout etc...

        self.backbone = Yolov8Backbone(scale=scale)
        self.head = YOLOv8Head(scale = scale)
        self.detection_head = Detectv8(nc = nc,
                                       ch = ch)

    def forward(self, x):

        # 1. Vision Features
        p3, p4, p5 = self.backbone(x)
        small, medium, large = self.head(p3, p4, p5)

        # 2. Detection
        # The detection head fuses the image features with the projected text features
        return self.detection_head([small, medium, large])


class YOLO11(nn.Module):
    def __init__(self,nc,scale,*args,**kwargs):
        super().__init__(*args, **kwargs)

        scale_map = {
            'n': [0.5, 0.25, 1024],
            's': [0.5, 0.5, 1024],
            'm': [0.5, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }

        # Extract depth (d), width (w), and max_channels (max_c)
        _, w, max_c = scale_map[scale]

        ch = (int(min(256 * w, max_c)),int(min(512 * w, max_c)),int(min(1024 * w, max_c)))            

        self.args = Arguments()###Arguments class set hypermeremeters like lambdas for loss function components,dropout etc...
        
        self.backbone = Yolo11Backbone(scale = scale)
        self.head = YOLO11Head(scale = scale)

        self.detection_head = Detect11(nc = nc,
                                       ch = ch)

    def forward(self, x):

        # 1. Vision Features
        p3, p4, p5 = self.backbone(x)
        small, medium, large = self.head(p3, p4, p5)

        # 2. Detection
        # The detection head fuses the image features with the projected text features
        return self.detection_head([small, medium, large])

class YOLOv26(nn.Module):
    def __init__(self,nc,scale,*args,**kwargs):
        super().__init__(*args, **kwargs)

        scale_map = {
            'n': [0.5, 0.25, 1024],
            's': [0.5, 0.5, 1024],
            'm': [0.5, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }

        # Extract depth (d), width (w), and max_channels (max_c)
        _, w, max_c = scale_map[scale]

        ch = (int(min(256 * w, max_c)),int(min(512 * w, max_c)),int(min(1024 * w, max_c)))     

        self.args = Arguments()###Arguments class set hypermeremeters like lambdas for loss function components,dropout etc...

        self.backbone = Yolov26Backbone(scale = scale)
        self.head = YOLOv26Head(scale = scale)
        ###YOLO26 use no dfl and is nms-free ,so end2end must be True
        self.detection_head = Detectv26(nc = nc,
                                        reg_max = 1,
                                        end2end = True,
                                        ch = ch)
    def forward(self, x):

        # 1. Vision Features
        p3, p4, p5 = self.backbone(x)
        small, medium, large = self.head(p3, p4, p5)

        # 2. Detection
        return self.detection_head([small, medium, large])    
    

class YOLOE(nn.Module):
    def __init__(self, text_model_name: str = None, class_names: list = [],scale:str = 'l',ch:tuple[int]= (256, 512, 512)):
        """
        params:
        text_model_name - string which contains text encoder models name(clip or mobileclip or something else)
        class_names - list of strings which contains object names which user whish to detect
        scale - model scale:large,small ,...
        ch: number of channels of p3,p4,p5 
        """
        
        super().__init__()
        

        ###check if there is text_model_name and class_names is not empty

        if text_model_name!=None and len(class_names)>0:
        
            # ============================================================
            # STEP 1: OFFLINE EMBEDDING CALCULATION
            # ============================================================
            print(f"Initializing Offline Embeddings for {len(class_names)} classes...")

            # A. Build the model temporarily
            temp_text_model = build_text_model(text_model_name)
            
            #2. Identify the device where the text model loaded itself (likely CUDA if available)
            # device = next(temp_text_model.parameters()).device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Some wrappers don't have a .to() method. We use a try-except just in case.
            try:
                temp_text_model = temp_text_model.to(device)
            except AttributeError:
                pass # Model is likely already frozen/handled internally

            # 3. Tokenize the text
            # This usually returns a CPU Tensor or a Dictionary of CPU Tensors
            text_token = temp_text_model.tokenize(class_names)

            # 4. CRITICAL FIX: Move token(s) to the same device as the model
            if isinstance(text_token, torch.Tensor):
                text_token = text_token.to(device)
            elif isinstance(text_token, dict):
                # Handle cases where tokenizer returns {'input_ids': ..., 'attention_mask': ...}
                text_token = {k: v.to(device) for k, v in text_token.items()}
            
            with torch.no_grad():
                raw_embeddings = temp_text_model.encode_text(text_token)
                
            # C. Reshape to [1, N_Classes, Embed_Dim] for broadcasting
            # Example: [1, 80, 512]
            raw_embeddings = raw_embeddings.reshape(1, len(class_names), -1)

            # =========================================================
            # THE FIX: Strip the "Inference Mode" tag
            # =========================================================
            # .detach() -> Disconnect from MobileCLIP graph
            # .clone()  -> Allocate NEW memory (creates a clean, standard tensor)
            clean_embeddings = raw_embeddings.detach().clone()
            
            # Register as a Buffer
            # - Buffers are saved in the state_dict (checkpoint),but if set persistent parameter False then it will not.
            # - Buffers are automatically moved to GPU with model.cuda()
            # - Buffers do NOT get updated by the optimizer
            self.register_buffer('offline_embeddings', clean_embeddings,persistent=False)
            
            #Get embedding dimension before deleting the model
            embed_dim = clean_embeddings.shape[-1]
            
            # DELETE the text model to save VRAM
            del temp_text_model
        
        #entry arguments
        self.args = Arguments()

        # ============================================================
        #BUILDING YOLO COMPONENTS

        self.backbone = Yolov8Backbone()
        self.head = YOLOv8Head()
        
        # We use len(class_names) as the number of classes (nc)
        self.detection_head = YOLOEv8Detect(nc=len(class_names), 
                                          ch=ch, ###change 1024->512 to avoid mismatch 
                                          with_bn=True, 
                                          embed=embed_dim)

    def forward(self, x):
        """
        Forward pass using cached (offline) text embeddings.
        """
        # 1. Vision Features
        p3, p4, p5 = self.backbone(x)
        small, medium, large = self.head(p3, p4, p5)

        # 2. Text Projection (The "Online" Learnable Part)

        ###check if repra's parameters was fused with classification head,then pass raw embeddings(taken from mobileclip output)
        if self.detection_head.is_fused:
            return self.detection_head([small, medium, large,self.offline_embeddings])

        ##else we must forward it trough reprta.

        # 3. Detection

        # We take the fixed offline embeddings and pass them through 
        # the learnable projection layers (get_tpe)
        # Input: [1, N, 512] (Fixed) -> Output: [1, N, Projected_Dim] (Learned)
        projected_embeddings = self.detection_head.get_tpe(self.offline_embeddings)

        # The detection head fuses the image features with the projected text features
        return self.detection_head([small, medium, large, projected_embeddings])

    def set_new_classes(self, new_class_names: list, text_model_name: str = "mobileclip:s0"):
        """
        Optional: Call this if you want to switch classes during Inference.
        This re-loads the text model briefly to calculate new embeddings.
        """
        temp_model = build_text_model(text_model_name).to(self.offline_embeddings.device)
        tokens = temp_model.tokenize(new_class_names).to(self.offline_embeddings.device)
        
        with torch.no_grad():
            new_embeddings = temp_model.encode_text(tokens)
            
        self.offline_embeddings = new_embeddings.reshape(1, len(new_class_names), -1)
        
        # Update detection head's nc (number of classes) if necessary
        self.detection_head.nc = len(new_class_names)
        del temp_model

    def fuse(self):
        self.detection_head.fuse(txt_feats=self.offline_embeddings)





class Arguments:
    def __init__(self,box = 7.5,cls = 0.5, dfl = 1.5, overlap_mask = True,mask_ratio = 4,dropout = 0.0):
        self.box = box
        self.cls = cls
        self.dfl = dfl
        self.overlap_mask = overlap_mask
        self.mask_ratio = mask_ratio
        self.dropout = dropout            


class YOLOEv8(YOLOE):
    def __init__(self, text_model_name = None, class_names = [], scale = 'l', ch = (256, 512, 512)):
        super().__init__(text_model_name, class_names, scale, ch)
        embed_dim = self.offline_embeddings.shape[-1]
        self.detection_head =   YOLOEv8Detect(nc=len(class_names), 
                                          ch=ch, ###change 1024->512 to avoid mismatch 
                                          with_bn=True, 
                                          embed=embed_dim)  


class YOLOE11(YOLOE):
    def __init__(self, text_model_name = None, class_names = [], scale = 'l'):

        scale_map = {
            'n': [0.5, 0.25, 1024],
            's': [0.5, 0.5, 1024],
            'm': [0.5, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        
        # Extract depth (d), width (w), and max_channels (max_c)
        _, w, max_c = scale_map[scale]

        ch = (int(min(256 * w, max_c)),int(min(512 * w, max_c)),int(min(1024 * w, max_c)))
        super().__init__(text_model_name, class_names, scale, ch = ch)

        self.backbone = Yolo11Backbone(scale=scale)
        self.head = YOLO11Head(scale = scale)
        ###take embedding dimension

        embed_dim = self.offline_embeddings.shape[-1]
        self.detection_head =   YOLOE11Detect(nc=len(class_names), 
                                          ch=ch, ###change 1024->512 to avoid mismatch 
                                          with_bn=True, 
                                          embed=embed_dim)


class YOLOEv26(YOLOE):
    def __init__(self, text_model_name = "mobileclip2:b", class_names = [], scale = 'l'):

        scale_map = {
            'n': [0.5, 0.25, 1024],
            's': [0.5, 0.5, 1024],
            'm': [0.5, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        
        # Extract depth (d), width (w), and max_channels (max_c)
        _, w, max_c = scale_map[scale]

        ch = (int(min(256 * w, max_c)),int(min(512 * w, max_c)),int(min(1024 * w, max_c)))
        super().__init__(text_model_name, class_names, scale, ch = ch)

        self.backbone = Yolov26Backbone(scale=scale)
        self.head = YOLOv26Head(scale = scale)
        ###take embedding dimension

        embed_dim = self.offline_embeddings.shape[-1]
        self.detection_head =   YOLOEv26Detect(nc=len(class_names), 
                                          ch=ch, ###change 1024->512 to avoid mismatch 
                                          with_bn=True, 
                                          embed=embed_dim,
                                          reg_max=1,
                                          end2end=True)            




    

if __name__ == "__main__":
    image_path = Path('datasets/skadi_in_yolo_format/images/0a0ce1aa0a0a880202067607160b5dd6_.jpg')
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(dim = 0)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    ###YOLOE(device) is to sfinaly move or keep txt_feats in correct device in text encoder
    image = image.to(device=device)
    names = ['car','van','minivan']
    model = YOLOEv26(class_names=names,scale='l').to(device)
    #model.load_state_dict(torch.load("models/yoloe-v8l-statedict.pt"))
    # prompt_embeddings = model.text_encoder(names)

    model.train()
    y = model(image)
    print(y.shape)
    


    




