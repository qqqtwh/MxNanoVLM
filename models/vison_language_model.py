import torch
import torch.nn as nn
from data.processors import get_tokenizer
from .vision_transformer import ViT
from .language_model import LanguageModel
from .modality_projector import ModalityProjector


class VisionLanguageModel(nn.Module):
    def __init__(self,cfg, backbone=True):
        super().__init__()
        self.cfg = cfg
        self.load_backbone_weights = backbone

        if self.load_backbone_weights:
            print("加载VLM预训练权重")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template, cfg.lm_cache_dir)
    
    def forward(self,input_ids, images, attention_mask=None, targets=None):
        pass

        return 1,torch.tensor([1.0],requires_grad=True)



    @classmethod
    def from_pretrained(cls) -> "VisionLanguageModel":
        pass

    def save_pretrained(self) -> None:
        pass

    def push_to_hub(self,) -> None:
        pass