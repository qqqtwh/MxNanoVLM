'''
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 train.py
'''

import os
import sys
os.chdir(sys.path[0])
import torch
import contextlib
from utils.torch_utils import init_dist,is_master,is_dist,destroy_dist,get_world_size,dist_gather,get_rank
from utils.common import load_config,get_run_name,get_lr
from utils.log import get_logger
from utils.evaluation import cli_evaluate
from data.dataloader import get_dataloaders
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import time
from statistics import mean
import argparse
from models.vison_language_model import VisionLanguageModel
from data.dataset import synchronized_dataloader_step

main_logger = get_logger('train.log')

def train(train_config,vlm_config):
    if is_master(): 
        main_logger.info('### 加载数据集与模型 ###')
    
    # 1.获取数据集与模型
    train_loader,val_loader = get_dataloaders(train_config, vlm_config)

    if train_config.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_config.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_config, backbone = vlm_config.vlm_load_backbone_weights)
    run_name = get_run_name(train_config,vlm_config)

    if is_master():
        main_logger.info(f"nanVLM 使用 {sum(p.numel() for p in model.parameters())} 参数量进行初始化")
        main_logger.info(f"训练摘要 {'(global)' if is_dist() else ''}: 在 {int(get_world_size()) if is_dist() else 1} 个GPU上训练")
        main_logger.info(f"     每个 GPU上 训练集有 {int(len(train_loader))} 个 batch, 验证集有 {int(len(val_loader))} 个 batch")
        main_logger.info(f"     batch_size {int(train_config.batch_size)}")
        main_logger.info(f"     梯度累积 {int(train_config.batch_size*train_config.gradient_accumulation_steps)} 个 batch 更新")
        main_logger.info('### 设置训练参数 ###')
    # 2.设置训练参数
    # 2.1.不同层设置不同学习率
    param_groups = [{
        'params': list(model.MP.parameters()),
        'lr': train_config.lr_mp
    }]

    if train_config.lr_backbones > 0:
        param_groups.append({
            'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 
            'lr': train_config.lr_backbones
        })
    else:
        for p in list(model.decoder.parameters()) + list(model.vision_encoder.parameters()):
            p.requires_grad = False

    # 2.2.优化器
    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in param_groups for p in group['params']]

    # 2.3.模型分配
    device = (
        torch.device('cuda') if torch.cuda.is_available() 
        else torch.device('mps') if hasattr(torch.backends,'mps') and torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    if device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    if is_master(): 
        main_logger.info(f'使用GPU设备: {device}')
    model.to(device)
    if train_config.compile:
        model = torch.compile(model)
    if is_dist():
        model = DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()])

    # 2.4.记录参数初始化
    epoch_times = []    # 所有 epoch 训练时长
    best_val_loss = float('inf') # 最好的验证损失
    global_step = 0     # 训练时梯度更新的次数
    epoch = 0           # 训练的 epoch，某些 epoch 不会更新，所以 global_step < epoch

    accumulated_stats = {
        'data_load_time': [],       # 单个 batch 数据的加载耗时
        'fw_bw_time': [],           # 单个 batch 前向传播+后向传播耗时
        'post_process_time': [],    # 单个 batch 梯度更新后处理耗时
        'tokens_per_second': [],    # 单 batch 所有GPU每秒推理的token数
        'images_per_sample': [],    # 单 batch 每个样本的图片数量
    }

    # 3.开始训练
    if is_master(): 
        main_logger.info('### 开始训练 ###')
    while global_step < train_config.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        total_train_loss = 0            # 单 epoch 训练损失
        total_tokens_processed = 0      # 单 epoch 推理的token总数
        data_load_start = time.time()

        model.train()
        optimizer.zero_grad()

        for i,batch in enumerate(synchronized_dataloader_step(train_loader, is_dist())):
            if is_master():
                main_logger.info(f'### {get_rank()} batch {i}/{len(train_loader)} ###')
                
            is_update_step = (i+1) % train_config.gradient_accumulation_steps or i+1 == len(train_loader)
            batch_start_time = time.time()

            images = batch['images']
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            idx = batch["idx"]

            data_load_end = time.time() 
            data_load_time = data_load_end - data_load_start

            # DDP中控制梯度同步时机的优化技巧，避免在梯度累积的中间步骤进行不必要的梯度同步，从而提升训练效率。
            if is_dist() and train_config.gradient_accumulation_steps>1 and not is_update_step:
                context = model.no_sync()   # 在该上下文内 loss.backward() 不会处罚梯度在多卡之间的同步
            else:
                context = contextlib.nullcontext() # 正常梯度同步

            fw_bw_start = time.time()
            autocast_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type in ['cuda','cpu'] else torch.float16)
            with autocast_context:
                with context:
                    _,loss = model(input_ids,images,attention_mask=attention_mask, targets=labels,idx=idx)
            if train_config.gradient_accumulation_steps>1: # 让每次 backward 的梯度规模平滑，便于稳定裁剪和优化
                loss = loss/train_config.gradient_accumulation_steps
            loss.backward()
            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                if train_config.max_grad_norm is not None: # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params,max_norm=train_config.max_grad_norm)
                
                # 动态更新 MP 层的 lr
                adj_lr_mp = get_lr(global_step, train_config.lr_mp, train_config.max_training_steps)
                optimizer.param_groups[0]['lr'] = adj_lr_mp
                
                # 动态更新 decoder 和 vision_encoder 层的 lr
                if train_config.lr_backbones > 0:
                    adj_lr_backbones = get_lr(global_step, train_config.lr_backbones, train_config.max_training_steps)
                    optimizer.param_groups[1]['lr'] = adj_lr_backbones

                # 梯度更新
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_config.gradient_accumulation_steps > 1:    # 等效完整 batch 的真实损失值
                batch_loss = batch_loss * train_config.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size()*num_tokens / batch_duration

            accumulated_stats['data_load_time'].append(data_load_time)
            accumulated_stats['tokens_per_second'].append(tokens_per_second)
            accumulated_stats['fw_bw_time'].append(fw_bw_time)
            accumulated_stats['post_process_time'].append(post_process_time)
            accumulated_stats['images_per_sample'].extend(images_per_sample)

            # 4.模型评估
            if train_config.eval_in_epochs and global_step % train_config.eval_interval == 0 and is_update_step and global_step > 0:
                if is_master(): 
                    main_logger.info('### 模型评估 ###')
                model.eval()
                if device == 'cuda':torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
                        total_val_loss += loss.item()

                    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            save_model = model.module if is_dist() else model
                            save_model.save_pretrained(save_directory=os.path.join(vlm_config.vlm_checkpoint_path, run_name))
                    
                    lmms_results = {}
                    if train_config.use_lmms_eval:
                        
                        
                        eval_args = argparse.Namespace(
                            model=model.module if is_dist() else model,
                            tasks=train_config.lmms_eval_tasks,
                            limit=train_config.lmms_eval_limit,
                            batch_size=train_config.lmms_eval_batch_size,
                            process_with_media=True,
                            device=device,
                        )
                        # Evaluate using the CLI wrapper
                        eval_results = cli_evaluate(eval_args,main_logger)

                        if is_master() and eval_results and "results" in eval_results[0]:
                            for task_name, task_results in eval_results[0]["results"].items():
                                for metric_name, metric_value in task_results.items():
                                    if isinstance(metric_value, (int, float)):
                                        lmms_results[f"{task_name}_{metric_name.split(',')[0]}"] = metric_value

                    if is_master():
                        main_logger.info(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
                model.train()        

            # 5.记录日志
            if global_step % train_config.stats_log_interval == 0 and len(accumulated_stats['tokens_per_second']) > 0 and is_update_step:
                stats = {}
                for key in ['tokens_per_second', 'data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]  # Flatten list of lists
                        stats[f'avg_{key}'] = mean(all_values_flat)
                    else:
                        stats[f'avg_{key}'] = mean(accumulated_stats[key])

                for key in ['data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f'max_{key}'] = max(all_values_flat)
                    else:
                        stats[f'max_{key}'] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(accumulated_stats['images_per_sample'])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats['min_images_per_sample'] = min(all_images_flat)
                else:
                    stats['min_images_per_sample'] = min(accumulated_stats['images_per_sample'])
                
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            if is_update_step:
                # 所有GPU上的平均 batch 损失
                batch_loss_gathered = mean(dist_gather(batch_loss)) if is_dist() else batch_loss

                global_step+=1
                if global_step>= train_config.max_training_steps:
                    break

            data_load_start = time.time()
            
        # 当前 epoch 所有GPU上的平均batch损失
        avg_train_loss = total_train_loss / len(train_loader) # 每个batch的平均损失
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # 当前 epoch 所有GPU上的平均每秒推理的 token 数
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            
            main_logger.info(f"Epoch: {epoch}, Step: {global_step}/{train_config.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

    # 6.输出日志
    if is_master():
        main_logger.info('### 输出日志 ###')
        total_training_time = sum(epoch_times)  # 总训练时长
        avg_train_time_per_epoch = total_training_time/len(epoch_times) # 每个epoch平均时长
        # 所有GPU上参与训练的总 batch_size
        batch_size = int(train_config.batch_size*get_world_size() * train_config.gradient_accumulation_steps)
        # 参与训练的样本总数(有重复)
        total_samples_processed =  batch_size * global_step
        # 平均每条样本的训练时长
        avg_time_per_sample = total_training_time / total_samples_processed
        main_logger.info(f"每个 epoch 平均时长: {avg_train_time_per_epoch:.2f}s")
        main_logger.info(f"每条样本的平均训练时长: {avg_time_per_sample:.4f}s")


def main():

    vlm_config = load_config('cfg/vlm.yaml')
    train_config = load_config('cfg/train.yaml')

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        main_logger.info('##################################')
        main_logger.info('### 配置信息 ###')
        main_logger.info('--- VLM 配置 ---')
        # main_logger.info(vlm_config)
        main_logger.info('--- Train 配置 ---')
        main_logger.info(train_config)
        
    
    train(train_config,vlm_config)

    if is_dist():
        destroy_dist()

if __name__ == '__main__':
    main()

    