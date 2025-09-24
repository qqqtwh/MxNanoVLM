## 项目说明
### 复现nanoVLM，并作出少量修改
- 修改数据加载逻辑，顺序读取 batch_size 条数据，适合个人低配 GPU 学习
- 修改少许代码结构，使代码更易读
### 结构
    数据集说明/
    assets/image.png        # 推理图像
    cfg/
        train.yaml          # 训练参数配置
        vlm.yaml            # 大模型参数配置
    checkpoints             # 模型保存路径
        /train_1
            config.json
            model.safetensors
    data/                   # 数据预处理模块代码
    models/                 # 大模型模块代码
    utils/                  # 工具类模块代码
    resources/              # 大模型与训练数据下载存放路径
        datasets/
            HuggingFaceM4___the_cauldron/
                ai2d
        models/
            google-siglip2-base-patch16-512/
            models--HuggingFaceTB--SmolLM2-360M-Instruct
    train.py                # 训练代码
    generate.py             # 推理代码

## 使用
### 训练
- cfg/文件夹中修改微调训练参数与模型参数
- 单卡训练 ```python train.py```
- 多卡训练 ```CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 train.py```
### 推理
- ```python generate.py```
### 评估



