from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f,SPPF,C3k2,C2PSA
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from PIL import Image

class Yolov8Backbone(nn.Module):
    def __init__(self, scale: str = "l"):
        super().__init__()

        scale_map = {
            'n': [0.33, 0.25, 1024],
            's': [0.33, 0.5, 1024],
            'm': [0.67, 0.75, 768],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.25, 512]
        }
        
        # Extract depth (d), width (w), and max_channels (max_c)
        d, w, max_c = scale_map[scale]

        # P1 & P2
        self.stem = nn.Sequential(
            Conv(3, int(min(64 * w, max_c)), 3, 2), 
            Conv(int(min(64 * w, max_c)), int(min(128 * w, max_c)), 3, 2),
            C2f(int(min(128 * w, max_c)), int(min(128 * w, max_c)), n=max(round(3 * d), 1), shortcut=True)
        )
        
        # P3 - Feature Map 1 (Scale 1/8)
        self.stage2 = nn.Sequential(
            Conv(int(min(128 * w, max_c)), int(min(256 * w, max_c)), 3, 2),
            C2f(int(min(256 * w, max_c)), int(min(256 * w, max_c)), n=max(round(6 * d), 1), shortcut=True)
        )
        
        # P4 - Feature Map 2 (Scale 1/16)
        self.stage3 = nn.Sequential(
            Conv(int(min(256 * w, max_c)), int(min(512 * w, max_c)), 3, 2),
            C2f(int(min(512 * w, max_c)), int(min(512 * w, max_c)), n=max(round(6 * d), 1), shortcut=True)
        )
        
        # P5 - Feature Map 3 (Scale 1/32)
        # The base channels here are 1024, but max_c will automatically cap them to 512 for 'l' and 'x' scales!
        self.stage4 = nn.Sequential(
            Conv(int(min(512 * w, max_c)), int(min(1024 * w, max_c)), 3, 2),
            C2f(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)), n=max(round(3 * d), 1), shortcut=True)
        )
        
        # SPPF
        self.sppf = SPPF(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)), 5)

    def forward(self, x):
        p2 = self.stem(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        out = self.sppf(p5)
        
        # Return a list/tuple for the Neck (FPN/PAN) to use
        return p3, p4, out


import torch.nn as nn


class Yolo11Backbone(nn.Module):
    def __init__(self, scale: str = 'l'):
        super().__init__()

        scale_map = {
            'n': [0.5, 0.25, 1024],
            's': [0.5, 0.5, 1024],
            'm': [0.5, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        
        # Extract depth (d), width (w), and max_channels (max_c)
        d, w, max_c = scale_map[scale]
        
        ### P1 and P2
        self.stem = nn.Sequential(
            Conv(3, int(min(64 * w, max_c)), 3, 2),
            Conv(int(min(64 * w, max_c)), int(min(128 * w, max_c)), 3, 2),
            C3k2(int(min(128 * w, max_c)), int(min(256 * w, max_c)), max(round(2 * d), 1), True, 0.25)
        )
        #P3
        self.stage2 = nn.Sequential(
            Conv(int(min(256 * w, max_c)), int(min(256 * w, max_c)), 3, 2),
            C3k2(int(min(256 * w, max_c)), int(min(512 * w, max_c)), max(round(2 * d), 1), True, 0.25)
        )
        #P4
        self.stage3 = nn.Sequential(
            Conv(int(min(512 * w, max_c)), int(min(512 * w, max_c)), 3, 2),
            C3k2(int(min(512 * w, max_c)), int(min(512 * w, max_c)), max(round(2 * d), 1), True)
        )
        #P5
        self.stage4 = nn.Sequential(
            Conv(int(min(512 * w, max_c)), int(min(1024 * w, max_c)), 3, 2),
            C3k2(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)), max(round(2 * d), 1), True)
        )
        
        ### SPPF 
        self.sppf = SPPF(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)), 5)
        
        ### C2PSA
        self.c2psa = C2PSA(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)),n=max(round(2 * d), 1))

    def forward(self,x):
        p2 = self.stem(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        out = self.sppf(p5)
        out = self.c2psa(out)

        return p3,p4,out
    

class Yolov26Backbone(nn.Module):
    def __init__(self,scale = 'l', *args, **kwargs):
        super().__init__(*args, **kwargs)
        scale_map = {
            'n': [0.5, 0.25, 1024],
            's': [0.5, 0.5, 1024],
            'm': [0.5, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        
        # Extract depth (d), width (w), and max_channels (max_c)
        d, w, max_c = scale_map[scale]

        ### P1 and P2
        self.stem = nn.Sequential(
            Conv(3, int(min(64 * w, max_c)), 3, 2),
            Conv(int(min(64 * w, max_c)), int(min(128 * w, max_c)), 3, 2),
            C3k2(int(min(128 * w, max_c)), int(min(256 * w, max_c)), max(round(2 * d), 1), True, 0.25)
        )

        #P3
        self.stage2 = nn.Sequential(
            Conv(int(min(256 * w, max_c)), int(min(256 * w, max_c)), 3, 2),
            C3k2(int(min(256 * w, max_c)), int(min(512 * w, max_c)), max(round(2 * d), 1), True, 0.25)
        )

        #P4
        self.stage3 = nn.Sequential(
            Conv(int(min(512 * w, max_c)), int(min(512 * w, max_c)), 3, 2),
            C3k2(int(min(512 * w, max_c)), int(min(512 * w, max_c)), max(round(2 * d), 1), True)
        )

        #P5
        self.stage4 = nn.Sequential(
            Conv(int(min(512 * w, max_c)), int(min(1024 * w, max_c)), 3, 2),
            C3k2(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)), max(round(2 * d), 1), True)
        )
        ### SPPF 
        self.sppf = SPPF(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)), 5,3,shortcut=True)
        
        ### C2PSA
        self.c2psa = C2PSA(int(min(1024 * w, max_c)), int(min(1024 * w, max_c)),n=max(round(2 * d), 1))


    def forward(self,x):
        p2 = self.stem(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        out = self.sppf(p5)
        out = self.c2psa(out)

        return p3,p4,out    
        




if __name__ == "__main__":
    image_path = Path('datasets/skadi_in_yolo_format/images/0a0ce1aa0a0a880202067607160b5dd6_.jpg')
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(dim = 0)

    model = Yolov26Backbone(scale='l')
    y = model(image)
    print("end of backbine test\n")
    # print(y.shape)