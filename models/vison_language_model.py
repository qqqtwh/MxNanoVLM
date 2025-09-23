import torch
import torch.nn as nn
from data.processors import get_tokenizer
from .vision_transformer import ViT
from .language_model import LanguageModel
from .modality_projector import ModalityProjector
import numpy as np
import torch.nn.functional as F

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
        
    
    def forward(self,input_ids, images, attention_mask=None, targets=None,idx=None):
        
        if isinstance(images,list):
            if not images:
                images = torch.empty(0,3, 512,512,device=input_ids.device)
            else:
                if isinstance(images[0],list):
                    images = [img for sublist in images for img in sublist]
                images = torch.cat(images,dim=0).to(input_ids.device)
                
        # 1.获取图像特征并将其进行模态转换
        image_bemd = self.vision_encoder(images)    # (bs,3,512,512) -> (bs, 32*32, 768)
        image_bemd = self.MP(image_bemd)            # (bs, 32*32, 768) ->(bs, (32//4)**2, 960)

        # 2.获取文本token的特征向量 -> token_embed
        token_embed = self.decoder.token_embedding(input_ids)   # (bs, 1024) -> (bs, 1024, 960)

        # 3.将 文本特征 中使用<|image|>标记的960维向量替换为对应的 960维图像特征
        updated_token_embd = self._replace_img_tokens_with_embd(input_ids,token_embed,image_bemd) # (bs, 1024, 960)

        # 4.LLM模型推理
        logits,_ = self.decoder(updated_token_embd, attention_mask=attention_mask) # (bs, 1024, 960)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits) # (bs, 1024, 49169)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    def _replace_img_tokens_with_embd(self,input_ids,token_embed,image_bemd):
        '''
            使用 image_bemd 替换 input_ids 中的 <|image|> 占位符
        '''
        update_token_emded = token_embed.clone()
        mask = (input_ids == self.tokenizer.image_token_id) # 找出 <|image|> 即 token_id = 49152的地方
        update_token_emded[mask] = image_bemd.view(-1, image_bemd.size(-1)).to(update_token_emded.dtype)

        return update_token_emded


    @classmethod
    def from_pretrained(cls) -> "VisionLanguageModel":
        pass

    def save_pretrained(self) -> None:
        pass

    def push_to_hub(self,) -> None:
        pass