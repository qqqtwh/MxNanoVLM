from .processors import get_image_processor,get_tokenizer
from utils.torch_utils import is_dist,get_world_size,get_rank,seed_worker,is_master
import torch
from torch.utils.data import DataLoader,DistributedSampler
from datasets import get_dataset_config_names,concatenate_datasets,load_dataset
from .dataset import VQADataset,ConstantLengthDataset
import torch.distributed as dist

class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def prepare_batch(self, batch, max_length=None):
        # [{},{},{}], 每个{}包含 "input_ids", "attention_mask", "labels", "images"
        # 不同样本的相同字段整合在一起
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference
        
        res = {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }
        return res

class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = batch[0]
        batch = self.prepare_batch(batch, max_length=self.max_length)
        
        return batch

def get_dataloaders(train_cfg,vlm_cfg):
    
    # 1.数据预处理器
    image_processor = get_image_processor(vlm_cfg.max_img_size,vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template,vlm_cfg.lm_cache_dir)

    # 2.下载梳理原始数据
    combined_train_data = []
    dataset_names_to_load = train_cfg.train_dataset_name
    if 'all' in dataset_names_to_load:
        try:
            dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path,local_files_only=True)
        except:
            print(f'{train_cfg.train_dataset_path} 下载失败，检测网络')
            
    
    for dataset_name in dataset_names_to_load:
        try:
            train_ds = load_dataset(
                train_cfg.train_dataset_path,
                dataset_name,
                cache_dir=train_cfg.train_dataset_cache,
            )
            combined_train_data.append(train_ds['train'])
        
        except Exception as e:
            if is_master():
                print(f"Warning: 该数据集加载失败 '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
            continue
    if not combined_train_data:
        raise ValueError('没有数据集被加载，请检查数据路径和名称。')
    
    train_ds = concatenate_datasets(combined_train_data).shuffle(seed=0)
    
    if is_dist():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # 设置具体使用多少条数据
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)
    else:
        total_samples = min(len(train_ds),train_cfg.data_cutoff_idx)
        print(f'只使用部分数据集 {total_samples}/{len(train_ds)}')
    
    # 3.划分
    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    g = torch.Generator()
    g.manual_seed(0)
    vqa_collator = VQACollator(tokenizer,vlm_cfg.lm_max_length)

    # 4.获得 datasets
    train_vqa_dataset = VQADataset(
        train_ds.select(range(train_size)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )
    val_vqa_dataset = VQADataset(
        train_ds.select(range(train_size,total_samples)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )

    train_dataset = ConstantLengthDataset(
        base_dataset = train_vqa_dataset,
        max_length = vlm_cfg.lm_max_length,
        batch_size = train_cfg.batch_size,
        rank = get_rank(),
        world_size = get_world_size()
    )

    val_dataset = ConstantLengthDataset(
        base_dataset = val_vqa_dataset,
        max_length = vlm_cfg.lm_max_length,
        batch_size = train_cfg.batch_size,
        rank = get_rank(),
        world_size = get_world_size()
    )
    
    train_loader = DataLoader(
        train_dataset,
        collate_fn=vqa_collator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    

    return train_loader, val_loader
