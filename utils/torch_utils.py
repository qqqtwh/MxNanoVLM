import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 强制使用单线程分词
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
import torch.distributed as dist
import numpy
import random

def init_dist():
    # 初始化分布式进程组
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    dist.barrier()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank()==0 if is_dist() else True

def destroy_dist():
    dist.destroy_process_group()

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all