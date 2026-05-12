from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f,C3k2
import torch
import torch.nn as nn

class YOLOv8Head(nn.Module):
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
        
        # --- Helper Functions for Clean Scaling ---
        def c(ch):
            """Calculates dynamic channels based on width and max_c limit."""
            return int(min(ch * w, max_c))
            
        def n(depth):
            """Calculates dynamic number of blocks based on depth."""
            return max(round(depth * d), 1)

        # Base feature map sizes coming from the Backbone
        p3_c = c(256)   # Stage 2 output
        p4_c = c(512)   # Stage 3 output
        p5_c = c(1024)  # SPPF output (Automatically becomes 512 for 'l' scale!)

        # =========================================================
        # TOP-DOWN PATH (FPN - Feature Pyramid Network)
        # =========================================================
        
        # 1. Upsample P5 and merge with P4
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Input: P5 + P4 -> Output: P4
        self.c2f_p4 = C2f(p5_c + p4_c, p4_c, n=n(3), shortcut=False)

        # 2. Upsample P4 and merge with P3
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Input: P4 + P3 -> Output: P3
        self.c2f_p3 = C2f(p4_c + p3_c, p3_c, n=n(3), shortcut=False)  # Small Object Feature

        # =========================================================
        # BOTTOM-UP PATH (PAN - Path Aggregation Network)
        # =========================================================
        
        # 3. Downsample P3 and merge with P4
        self.cv1 = Conv(p3_c, p3_c, 3, 2)
        
        # Input: Downsampled_P3 + P4 -> Output: P4
        self.c2f_medium = C2f(p3_c + p4_c, p4_c, n=n(3), shortcut=False)  # Medium Object Feature

        # 4. Downsample P4 and merge with P5
        self.cv2 = Conv(p4_c, p4_c, 3, 2)
        
        # Input: Downsampled_P4 + P5 -> Output: P5
        self.c2f_large = C2f(p4_c + p5_c, p5_c, n=n(3), shortcut=False)  # Large Object Feature

    def forward(self, p3, p4, p5):
        """
        p3: Backbone feature P3 (Scale 1/8) - 256 channels
        p4: Backbone feature P4 (Scale 1/16) - 512 channels
        p5: Backbone feature P5 (Scale 1/32) - 512 channels (SPPF output)
        """
        
        # --- Top-Down ---
        x = self.up1(p5)
        # Concat backbone P4 with upsampled P5
        x = torch.cat([x, p4], dim=1) 
        head_p4 = self.c2f_p4(x) # Save this (Layer 12) for later concatenation

        x = self.up2(head_p4)
        # Concat backbone P3 with upsampled Head_P4
        x = torch.cat([x, p3], dim=1)
        head_p3 = self.c2f_p3(x) # Layer 15 (Result for Small Objects)

        # --- Bottom-Up ---
        x = self.cv1(head_p3)
        # Concat Layer 12 (head_p4) with downsampled Head_P3
        x = torch.cat([x, head_p4], dim=1)
        out_medium = self.c2f_medium(x) # Layer 18 (Result for Medium Objects)

        x = self.cv2(out_medium)
        # Concat Backbone P5 (p5) with downsampled Out_Medium
        x = torch.cat([x, p5], dim=1)
        out_large = self.c2f_large(x) # Layer 21 (Result for Large Objects)

        return head_p3, out_medium, out_large
    


class YOLO11Head(nn.Module):
    def __init__(self,scale = 'l'):
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

        # =========================================================
        # TOP-DOWN PATH (FPN - Feature Pyramid Network)
        # =========================================================
        
        # 1. Upsample P5 (Layer 9) and merge with P4 (Layer 6)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # # Input: P5(1024) + P4(512) = 1536 channels -> Output: 512
        # self.c2f_p4 = C2f(1024 + 512, 512, n=3, shortcut=False) # Layer 12

        # 1. FIX: Input is P5(512) + P4(512) = 1024
        # (Was 1536)
        self.c3k2_p4 = C3k2(int(min(1024 * w, max_c)) + int(min(512 * w, max_c)), int(min(512 * w,max_c)), max(round(2 * d), 1),c3k=True,shortcut=False)

        # 2. Upsample P4 (Layer 12) and merge with P3 (Layer 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Input: Layer12(512) + P3(512) = 1025 channels -> Output: 256
        self.c3k2_p3 = C3k2(int(min(512 * w,max_c)) + int(min(512 * w,max_c)), int(min(256 * w,max_c)),max(round(2 * d),1) , c3k=True,shortcut=False) # Layer 15 (Small Object Feature)

        # =========================================================
        # BOTTOM-UP PATH (PAN - Path Aggregation Network)
        # =========================================================
        
        # 3. Downsample P3 (Layer 15) and merge with P4 (Layer 12)
        self.cv1 = Conv(int(min(256 * w,max_c)), int(min(256 * w,max_c)), 3, 2) # Stride 2 convolution
        
        # Input: Downsampled_P3(256) + Layer12(512) = 768 channels -> Output: 512
        self.c3k2_medium = C3k2(int(min(256 * w,max_c)) + int(min(512 * w,max_c)), int(min(512 * w,max_c)),max(round(2 * d), 1) ,c3k=True,shortcut=False) # Layer 18 (Medium Object Feature)

        # 4. Downsample P4 (Layer 18) and merge with P5 (Layer 9)
        self.cv2 = Conv(int(min(512 * w,max_c)), int(min(512 * w,max_c)), 3, 2) # Stride 2 convolution
        
        # Input: Downsampled_P4(512) + P5(1024) = 1536 channels -> Output: 1024
        # self.c2f_large = C2f(512 + 1024, 1024, n=3, shortcut=False) # Layer 21 (Large Object Feature)
        
        # 2. FIX: Input is P4(512) + P5(512) = 1024
        # (Was 1536)
        # Output stays 512 to match the backbone's max width
        self.c3k2_large = C3k2(int(min(1024 * w, max_c)) + int(min(512 * w, max_c)), int(min(1024 * w, max_c)), max(round(2 * d), 1),c3k = True ,shortcut=True)


    def forward(self,p3,p4,p5):

        """
        p3: Backbone feature P3 (Scale 1/8) - min( 256 * w, max_c) channels
        p4: Backbone feature P4 (Scale 1/16) - min( 512 * w, max_c) channels
        p5: Backbone feature P5 (Scale 1/32) - min( 1024 * w, max_c) channels (SPPF output)
        """

        # --- Top-Down ---
        x = self.up1(p5)
        # Concat backbone P4 with upsampled P5
        x = torch.cat([x, p4], dim=1) 
        head_p4 = self.c3k2_p4(x) # Save this (Layer 12) for later concatenation

        x = self.up2(head_p4)
        # Concat backbone P3 with upsampled Head_P4
        x = torch.cat([x, p3], dim=1)
        head_p3 = self.c3k2_p3(x) # Layer 15 (Result for Small Objects)

        # --- Bottom-Up ---
        x = self.cv1(head_p3)
        # Concat Layer 12 (head_p4) with downsampled Head_P3
        x = torch.cat([x, head_p4], dim=1)
        out_medium = self.c3k2_medium(x) # Layer 18 (Result for Medium Objects)

        x = self.cv2(out_medium)
        # Concat Backbone P5 (p5) with downsampled Out_Medium
        x = torch.cat([x, p5], dim=1)
        out_large = self.c3k2_large(x) # Layer 21 (Result for Large Objects)

        return head_p3, out_medium, out_large
    

class YOLOv26Head(nn.Module):
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

        # =========================================================
        # TOP-DOWN PATH (FPN - Feature Pyramid Network)
        # =========================================================
        
        # 1. Upsample P5 (Layer 9) and merge with P4 (Layer 6)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # # Input: P5(1024) + P4(512) = 1536 channels -> Output: 512
        # self.c2f_p4 = C2f(1024 + 512, 512, n=3, shortcut=False) # Layer 12

        # 1. FIX: Input is P5(1024) + P4(512) = 1536
        # (Was 1536)
        self.c3k2_p4 = C3k2(int(min(1024 * w, max_c)) + int(min(512 * w, max_c)), int(min(512 * w,max_c)), max(round(2 * d), 1),c3k=True,shortcut=False)

        # 2. Upsample P4 (Layer 12) and merge with P3 (Layer 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Input: Layer12(512) + P3(512) = 1024 channels -> Output: 256
        self.c3k2_p3 = C3k2(int(min(512 * w,max_c)) + int(min(512 * w,max_c)), int(min(256 * w,max_c)),max(round(2 * d),1) , c3k=True,shortcut=False) # Layer 15 (Small Object Feature)

        # =========================================================
        # BOTTOM-UP PATH (PAN - Path Aggregation Network)
        # =========================================================
        
        # 3. Downsample P3 (Layer 15) and merge with P4 (Layer 12)
        self.cv1 = Conv(int(min(256 * w,max_c)), int(min(256 * w,max_c)), 3, 2) # Stride 2 convolution
        
        # Input: Downsampled_P3(256) + Layer12(512) = 768 channels -> Output: 512
        self.c3k2_medium = C3k2(int(min(256 * w,max_c)) + int(min(512 * w,max_c)), int(min(512 * w,max_c)),max(round(2 * d), 1) ,c3k=True,shortcut=False) # Layer 18 (Medium Object Feature)

        # 4. Downsample P4 (Layer 18) and merge with P5 (Layer 9)
        self.cv2 = Conv(int(min(512 * w,max_c)), int(min(512 * w,max_c)), 3, 2) # Stride 2 convolution
        
        # Input: Downsampled_P4(512) + P5(1024) = 1536 channels -> Output: 1024
        # self.c2f_large = C2f(512 + 1024, 1024, n=3, shortcut=False) # Layer 21 (Large Object Feature)
        
        # 2. FIX: Input is P4(512) + P5(512) = 1024
        # (Was 1536)
        # Output stays 512 to match the backbone's max width
        self.c3k2_large = C3k2(int(min(1024 * w, max_c)) + int(min(512 * w, max_c)), int(min(1024 * w, max_c)), max(round(1 * d), 1),c3k = True,e=0.5 ,shortcut=True,attn=True)


    def forward(self,p3,p4,p5):

        """
        p3: Backbone feature P3 (Scale 1/8) - min( 256 * w, max_c) channels
        p4: Backbone feature P4 (Scale 1/16) - min( 512 * w, max_c) channels
        p5: Backbone feature P5 (Scale 1/32) - min( 1024 * w, max_c) channels (SPPF output)
        """

        # --- Top-Down ---
        x = self.up1(p5)
        # Concat backbone P4 with upsampled P5
        x = torch.cat([x, p4], dim=1) 
        head_p4 = self.c3k2_p4(x) # Save this (Layer 12) for later concatenation

        x = self.up2(head_p4)
        # Concat backbone P3 with upsampled Head_P4
        x = torch.cat([x, p3], dim=1)
        head_p3 = self.c3k2_p3(x) # Layer 15 (Result for Small Objects)

        # --- Bottom-Up ---
        x = self.cv1(head_p3)
        # Concat Layer 12 (head_p4) with downsampled Head_P3
        x = torch.cat([x, head_p4], dim=1)
        out_medium = self.c3k2_medium(x) # Layer 18 (Result for Medium Objects)

        x = self.cv2(out_medium)
        # Concat Backbone P5 (p5) with downsampled Out_Medium
        x = torch.cat([x, p5], dim=1)
        out_large = self.c3k2_large(x) # Layer 21 (Result for Large Objects)

        return head_p3, out_medium, out_large    