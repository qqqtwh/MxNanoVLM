import yaml
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
import os

@dataclass
class TrainConfig:
    # Training Hyperparameters
    lr_mp: float = 0.00512
    lr_backbones: float = 5e-5

    # Data Settings
    data_cutoff_idx: Optional[int] = None
    val_ratio: float = 0.025
    batch_size: int = 8
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 1024

    # Training Loop
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 800    # gradient_accumulation_steps * 100
    stats_log_interval: int = 200 # gradient_accumulation_steps * 25
    max_training_steps: int = 5000

    # Model & Compilation
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False

    # Dataset
    train_dataset_cache: str = "./resources/datasets"
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: Tuple[str, ...] = ("all",)

    # Logging & Evaluation
    log_wandb: bool = True
    wandb_entity: str = "HuggingFace"

    # lmms-eval Integration
    use_lmms_eval: bool = True
    lmms_eval_tasks: str = 'mmstar,mmmu,ocrbench,textvqa'
    lmms_eval_limit: int = 2000
    lmms_eval_batch_size: int = 128

    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainConfig':
        """从 YAML 文件加载配置并构造 TrainConfig 实例"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}

        # 获取所有字段名和默认值
        field_defaults = {f.name: f.default for f in cls.__dataclass_fields__.values() if f.default is not field}

        # 过滤掉 init=False 的字段（因为它们不能从构造函数传入）
        init_fields = {name: f for name, f in cls.__dataclass_fields__.items() if f.init}

        # 构造传入参数字典，缺失字段使用默认值
        kwargs = {}
        for name, field_info in init_fields.items():
            if name in config_dict:
                value = config_dict[name]
                # YAML 中的 null 映射为 None
                if value == 'null':
                    value = None
                # 类型转换（可选增强）
                if field_info.type is int and isinstance(value, str):
                    value = int(value)
                elif field_info.type is float and isinstance(value, str):
                    value = float(value)
                kwargs[name] = value
            else:
                # 使用默认值（包括 field(default=...)）
                if field_info.default is not field:
                    kwargs[name] = field_info.default
                elif field_info.default_factory is not field:
                    kwargs[name] = field_info.default_factory()

        # 创建实例
        instance = cls(**kwargs)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（可用于日志或调试）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class VLMConfig:
    # --- Vision Transformer (ViT) ---
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 768*4 # 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    max_img_size: int = 1024
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip2-base-patch16-512"
    vit_model_dir: str = "./resources/models/google-siglip2-base-patch16-512"

    # --- Language Model (LM) ---
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 17
    lm_vocab_size: int = 49152 + 17 # lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 1024
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True

    # --- ModalityProjector (MP) ---
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    # --- LM Token & Paths ---
    lm_cache_dir: str = "./resources/models"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template: str = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    vlm_extra_tokens: Dict[str, str] = field(default_factory=lambda: {
        "image_token": "<|image|>",
        "r1c1": "<row_1_col_1>",
        "r1c2": "<row_1_col_2>",
        "r1c3": "<row_1_col_3>",
        "r1c4": "<row_1_col_4>",
        "r2c1": "<row_2_col_1>",
        "r2c2": "<row_2_col_2>",
        "r2c3": "<row_2_col_3>",
        "r2c4": "<row_2_col_4>",
        "r3c1": "<row_3_col_1>",
        "r3c2": "<row_3_col_2>",
        "r3c3": "<row_3_col_3>",
        "r3c4": "<row_3_col_4>",
        "r4c1": "<row_4_col_1>",
        "r4c2": "<row_4_col_2>",
        "r4c3": "<row_4_col_3>",
        "r4c4": "<row_4_col_4>",
    })
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints"
    hf_repo_name: str = "nanoVLM"

    @classmethod
    def from_yaml(cls, config_path: str) -> 'VLMConfig':
        """从 YAML 文件加载配置并构造 VLMConfig 实例"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}

        # 获取所有可初始化字段
        init_fields = {name: f for name, f in cls.__dataclass_fields__.items() if f.init}

        kwargs = {}
        for name, field_info in init_fields.items():
            if name in config_dict:
                value = config_dict[name]
                if value == 'null':
                    value = None

                # 特殊处理：vlm_extra_tokens 应该是 dict
                if name == 'vlm_extra_tokens' and isinstance(value, dict):
                    pass  # 保持原样
                # 基础类型转换增强（可选）
                elif field_info.type is int and isinstance(value, str):
                    value = int(value)
                elif field_info.type is float and isinstance(value, str):
                    value = float(value)

                kwargs[name] = value
            else:
                # 使用默认值
                if field_info.default is not field:
                    kwargs[name] = field_info.default
                elif field_info.default_factory is not field:
                    kwargs[name] = field_info.default_factory()

        instance = cls(**kwargs)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于调试或日志"""
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, dict):
                result[k] = v.copy()
            else:
                result[k] = v
        return result