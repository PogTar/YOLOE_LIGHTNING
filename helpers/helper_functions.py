import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict

def compare_chkpts(ultrltcs_chkpt:str,custm_ckpt:str):
    """
    purpose of this helper function is to check was batchnorm parameters changed during training or not

    params:
    ultrltcs_chkpt: pt file provided by ultralytics
    custm_ckpt :pth file saved after finr_tuning
    """
    print(f"Loading weights from {ultrltcs_chkpt}...")
    
    # 1. Load the checkpoint
    ckpt = torch.load(ultrltcs_chkpt,weights_only=False)
    # Extract the state dict
    if hasattr(ckpt.get('model'), 'state_dict'):
        state_dict = ckpt['model'].state_dict()
    else:
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    # 2. Define the Mapping: { Old_Prefix : New_Prefix }
    # This maps 'model.X' -> 'backbone/head.layer_name'
    mapping = {
        # --- BACKBONE ---
        'model.0.': 'backbone.stem.0.',
        'model.1.': 'backbone.stem.1.',
        'model.2.': 'backbone.stem.2.',
        'model.3.': 'backbone.stage2.0.',
        'model.4.': 'backbone.stage2.1.',
        'model.5.': 'backbone.stage3.0.',
        'model.6.': 'backbone.stage3.1.',
        'model.7.': 'backbone.stage4.0.',
        'model.8.': 'backbone.stage4.1.',
        'model.9.': 'backbone.sppf.',
        
        # --- HEAD (Neck) ---
        # Note: Layers 10, 11, 13, 14, 17, 20 are Upsample/Concat (No weights), so we skip them
        'model.12.': 'head.c2f_p4.',
        'model.15.': 'head.c2f_p3.',
        'model.16.': 'head.cv1.',
        'model.18.': 'head.c2f_medium.',
        'model.19.': 'head.cv2.',
        'model.21.': 'head.c2f_large.',
        
        # --- DETECTION HEAD ---
        'model.22.': 'detection_head.'
    }

    # 3. Create a new State Dict with renamed keys
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in mapping.items():
            if key.startswith(old_prefix):
                # Replace model.0.conv... with backbone.stem.0.conv...
                new_key = key.replace(old_prefix, new_prefix)
                break
        
        new_state_dict[new_key] = value

    # 4. Filter out Shape Mismatches (Crucial for Transfer Learning)
    # If your model has nc=3 but checkpoint has nc=80, the final layers won't match.
    model_state_dict = torch.load(custm_ckpt)

    diff_bn_weights = list()
    for key, value in model_state_dict.items():
        
        if (not 'cv3' in key) and (not 'bn' in key):
            try:
                if not torch.equal(value,new_state_dict[key]):
                    print(f"{key} differs")
                    diff_bn_weights.append(key)
            except KeyError:
                print(f"no such key:{key} in ultralytics checkpoint file")      

    print(len(diff_bn_weights))


def create_mapping(yolo_version:str = 'v8')->dict[str,str]:
    """
    this function must correctly map module names from checkpoint file to my own structure
    based on version.Early I put this step inside load_yoloe_weights,but to have more readable and 
    compact code I create function which return version based mapping.

    params:
    yolo_version: version of backbone.Currently avaliable only v8 and 11 versions

    return:
    dectionary[key = checkpoint_name:value = custom_name] 
    """    

    if yolo_version == 'v8':
        # YOLOv8 has 23 layers (0-22)
        mapping = {
            'model.0.': 'backbone.stem.0.', 'model.1.': 'backbone.stem.1.', 'model.2.': 'backbone.stem.2.',
            'model.3.': 'backbone.stage2.0.', 'model.4.': 'backbone.stage2.1.',
            'model.5.': 'backbone.stage3.0.', 'model.6.': 'backbone.stage3.1.',
            'model.7.': 'backbone.stage4.0.', 'model.8.': 'backbone.stage4.1.',
            'model.9.': 'backbone.sppf.',
            # Neck/Head (v8 indices)
            'model.12.': 'head.c2f_p4.',
            'model.15.': 'head.c2f_p3.',
            'model.16.': 'head.cv1.',
            'model.18.': 'head.c2f_medium.',
            'model.19.': 'head.cv2.',
            'model.21.': 'head.c2f_large.',
            'model.22.': 'detection_head.'
        }
    elif yolo_version in ['11','v26']:
        # YOLO11 has 24 layers (0-23) because C2PSA is inserted at index 10
        mapping = {
            'model.0.': 'backbone.stem.0.', 'model.1.': 'backbone.stem.1.', 'model.2.': 'backbone.stem.2.',
            'model.3.': 'backbone.stage2.0.', 'model.4.': 'backbone.stage2.1.',
            'model.5.': 'backbone.stage3.0.', 'model.6.': 'backbone.stage3.1.',
            'model.7.': 'backbone.stage4.0.', 'model.8.': 'backbone.stage4.1.',
            'model.9.': 'backbone.sppf.',
            'model.10.': 'backbone.c2psa.',        # <--- NEW IN YOLO11
            # Neck/Head (v11 indices shifted by +1)
            'model.13.': 'head.c3k2_p4.',          # Was 12 in v8
            'model.16.': 'head.c3k2_p3.',          # Was 15 in v8
            'model.17.': 'head.cv1.',              # Was 16 in v8
            'model.19.': 'head.c3k2_medium.',      # Was 18 in v8
            'model.20.': 'head.cv2.',              # Was 19 in v8
            'model.22.': 'head.c3k2_large.',       # Was 21 in v8
            'model.23.': 'detection_head.'         # Was 22 in v8
        }
    else:
        raise ValueError("Unsupported version. Choose 'v8','11' or 'v26'.")
    
    return mapping


def load_yoloe_weights(model, weights_path:str|Path, backbone_version,strict,):
    print(f"Loading weights from {weights_path}...")
    
    # 1. Load the checkpoint
    ckpt = torch.load(weights_path, weights_only=False)
    if hasattr(ckpt.get('model'), 'state_dict'):
        state_dict = ckpt['model'].state_dict()
    elif isinstance(ckpt,dict) and 'state_dict' in ckpt:
        state_dict = OrderedDict([
    (k.removeprefix('model.'), v) for k, v in ckpt['state_dict'].items()
])
    else:
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    ###If weights path ends with .ckpt it means that this chekpoint is saved by me,so we don't need any mapping we must just load weights
    if not weights_path.endswith(".ckpt"):
        # 2. Define the Mapping
        mapping = create_mapping(yolo_version=backbone_version)

        # 3. Create a new State Dict with renamed keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix)
                    break
            new_state_dict[new_key] = value

        # 4. Filter and Translate Fused to Unfused
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for k, v in new_state_dict.items():
            # --- THE GENIUS HACK: Catch the fused bias and route it to the BN layer ---
            if k.endswith('.conv.bias'):
                bn_bias_key = k.replace('.conv.bias', '.bn.bias')
                if bn_bias_key in model_state_dict:
                    filtered_state_dict[bn_bias_key] = v
                else:
                    print(f"Skipping {k}: Couldn't map to a BN bias.")
                    
            # --- Standard weight matching ---
            elif k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping {k}: Shape mismatch {v.shape} vs {model_state_dict[k].shape}")
            else:
                print(f"Skipping {k}: Key not found in model.")

        # --- NEUTRALIZE MISSING BATCHNORM LAYERS ---
        # Since the fused checkpoint has no BN weights/means/vars, we MUST 
        # force them to 1.0 and 0.0, otherwise they stay initialized as random garbage!
        for k in model_state_dict.keys():
            if '.bn.weight' in k and k not in filtered_state_dict:
                filtered_state_dict[k] = torch.ones_like(model_state_dict[k])
            elif '.bn.running_mean' in k and k not in filtered_state_dict:
                filtered_state_dict[k] = torch.zeros_like(model_state_dict[k])
            elif '.bn.running_var' in k and k not in filtered_state_dict:
                filtered_state_dict[k] = torch.ones_like(model_state_dict[k])
            elif '.bn.bias' in k and k not in filtered_state_dict:
                # Just in case a layer didn't have a fused bias either
                filtered_state_dict[k] = torch.zeros_like(model_state_dict[k])

        # 5. Load the weights
        # strict=False allows us to skip the mismatched final classification layers
        model.load_state_dict(filtered_state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)   
    print(f"Weights loaded successfully (with strict={strict}).")

def prepare_model(model:nn.Module,weights_path:str,linear_probing:bool,backbone_version:str,only_cv3:bool = True,strict:bool = True):
    """
    params:
    linear_probing:if Linear probing is False then trainable can be only detection head,else hole model
    only_cv3:If True,then keep trainable only classification head if False then keep trainable hole 
             detection head.Default Value is True(Warning if only_cv3 = False then desirible to set small learning rates ~1e-5).
    """
    
    load_yoloe_weights(model = model,weights_path=weights_path,backbone_version=backbone_version,strict=strict)

    if linear_probing:
        # 1. First, freeze everything
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Specifically unfreeze the detection head weights
        if only_cv3:
            ###we can do here some modifications.
            trainable_modules:list[str] = ["detection_head.cv3","detection_head.one2one_cv3"]
        else:
            trainable_modules:list[str] = ["detection_head"]    
        for name, param in model.named_parameters():
            for trainable_module in trainable_modules:
                if trainable_module in name:
                    param.requires_grad = True

        # 3. Force BN/Norm layers into eval mode to freeze statistics
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                m.eval()
                # Ensure even if we call model.train() later, these stay in eval
                m.track_running_stats = False # Optional: strict freeze




# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    import os
    import sys

    # Get the path of the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the path of the parent directory
    parent_dir = os.path.join(current_dir, '..')

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    # Now you can import the module from the parent directory
    from modules.detect import YOLOv8# Replace 'module_in_parent' with your actual filename (without .py)

    model = YOLOv8(nc = 1,scale="l")
    ###delete savpe since we don't use visual prompts
    # del model.detection_head.savpe

    load_yoloe_weights(model, "models/yoloe-v8l-seg.pt",backbone_version='v8',strict=False)

    print("End of test")
    # compare_chkpts(ultrltcs_chkpt= "models/yoloe-v8l-seg-statedict.pt",custm_ckpt="models/yoloe_linear_probed_on_skadi_ctz_earlystpooing_smaller_lr.pth")


  