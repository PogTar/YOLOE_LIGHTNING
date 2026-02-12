import lightning as L
import ultralytics
import torch,torchvision,torchaudio
import numpy
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.utils import loss
from ultralytics.nn.text_model import build_text_model


class YoloeLighning(L.LightningModule):
    def __init__(self, *args,text_prompt):
        super.__init__()
        self.head = YOLOEDetect(*args)
        self.clip = build_text_model("mobileclip:s0") 
        self.text_embeddings = self.text_encoder() 
        self.loss = loss.v8DetectionLoss()

    def text_encoder(self,text_prompt:str)->torch.Tensor:
        text_token = self.clip.tokenize(text_prompt)
        txt_feats = self.clip.encode_text(text_token)
        txt_feats = txt_feats.reshape(-1, len(text_prompt), txt_feats.shape[-1])

        ###get normalized text embeddings
        return self.head.get_tpe(txt_feats)

