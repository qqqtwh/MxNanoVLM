import torch
import torch.nn as nn
from data.processors import get_tokenizer
from .vision_transformer import ViT
from .language_model import LanguageModel
from .modality_projector import ModalityProjector
import numpy as np
import torch.nn.functional as F
from utils.common import top_k_top_p_filtering
import os
from typing import Optional
from safetensors.torch import load_model, save_model
from data.config import VLMConfig
import json
from dataclasses import asdict

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
    def from_pretrained(cls, repo_id_or_path: str, *, revision: Optional[str] = None) -> "VisionLanguageModel":
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self,save_directory) -> None:
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self,repo_id: str, private: bool = False) -> None:
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)
        import tempfile
        MODEL_CARD_TEMPLATE = """
            ---
            library_name: mxnanovlm
            license: mit
            pipeline_tag: image-text-to-text
            tags:
            - vision-language
            - multimodal
            - research
            ---

            **Usage:**

            ```python
            from models.vision_language_model import VisionLanguageModel

            model = VisionLanguageModel.from_pretrained("{repo_id}")
            ```
            """

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )

    @torch.inference_mode()
    def generate(
        self,
        input_ids,              # 当前batch pad后的 token_ids
        images,                 # 当前batch预处理后的图像
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False            # 是否进行 top_k,top_p,temperature 处理结果
    ):
        
        if isinstance(images, list):
            if not images:
                images = torch.empty(0, 3, self.cfg.vit_image_size, self.cfg.vit_image_size, device=input_ids.device)
            else:
                if isinstance(images[0],list):
                    images = [img for sublist in images for img in sublist]

                images = torch.cat(images, dim=0).to(input_ids.device)
        
        # 1.初始推理，获得最后1个token的特征
        image_bemd = self.vision_encoder(images)
        image_bemd = self.MP(image_bemd)
        
        token_embed = self.decoder.token_embedding(input_ids)

        initial_combined_embeds = self._replace_img_tokens_with_embd(input_ids,token_embed,image_bemd)   # (bs, 1024, 960)

        current_total_seq_len = initial_combined_embeds.size(1) # 1024
        batch_size = input_ids.size(0)

        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )

        last_token_output_from_prefill = prefill_output[:, -1, :] #  (bs, 960)
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 
        
        # 2.根据最后1个token的特征，获得对应 token_id，连续 max_new_tokens 次
        newly_generated_ids_list = []

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)  #（bs, 1）
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)   # (bs, 960)
                probs = torch.softmax(filtered_logits/temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)  # （bs, 1）
                
            newly_generated_ids_list.append(next_token_id)
            next_token_embed = self.decoder.token_embedding(next_token_id) # (bs, 1, 960)

            current_token_start_pos = current_total_seq_len # 1024
            current_total_seq_len += 1 # 1025

            if attention_mask is not None: # (bs, 1024)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones((batch_size,1), device = attention_mask.device, dtype=attention_mask.dtype)),dim=1)
                # (bs, 1025)

            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask = attention_mask,
                kv_cache = kv_cache_list,
                start_pos = current_token_start_pos
            ) # (bs, 1, 960)

            last_token_output = decode_step_output[:,-1,:] # (bs, 960)

            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output         # (bs, 960)

        if not newly_generated_ids_list:
            return torch.empty((batch_size,0),dtype=torch.long,device=input_ids.device)
        
        # 3.后处理
        generated_ids = torch.cat(newly_generated_ids_list,dim=1) # (bs,5)
        if self.tokenizer.eos_token_id is not None and generated_ids.numel()>0:
            seq_len = generated_ids.size(1) # 5
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id)
            col_indices_for_min = torch.arange(seq_len, device=device) # 列

            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)

            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids