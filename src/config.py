from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    """
    训练配置类
    
    用于集中管理所有超参数，方便在不同脚本间共享配置。
    """
    
    # --- 1. 图像与模型参数 ---
    image_size: int = 64        # 输入图像分辨率 (64x64)，适合 Mac 快速训练
    patch_size: int = 4         # DiT 的 Patch 大小，将图像切分为 (64/4)x(64/4) = 16x16 个 tokens
    
    # DiT 模型规模 (Tiny 配置)
    num_layers: int = 6         # Transformer 层数
    num_attention_heads: int = 4 # 注意力头数
    attention_head_dim: int = 64 # 每个头的维度 (总 hidden_dim = 4 * 64 = 256)
    
    # --- 2. 训练参数 ---
    train_batch_size: int = 32  # 批次大小，CIFAR10 较小，可以适当增大 batch
    eval_batch_size: int = 32   # 验证/推理时的批次大小
    num_epochs: int = 5         # 训练轮数 (CIFAR10 数据量大，5 轮相当于 300k 样本)
    learning_rate: float = 1e-4 # 学习率
    lr_warmup_steps: int = 500  # 学习率预热步数
    
    # --- 3. 混合精度与加速 ---
    # Mac MPS 对 fp16 的支持已改善，但在某些 accelerate 版本中仍有限制
    # 如果报错 "fp16 mixed precision requires a GPU"，请回退到 "no"
    mixed_precision: str = "no" 
    
    # DataLoader 优化
    dataloader_num_workers: int = 4  # Mac M系列芯片通常有多个核心，开启多进程加载
    dataloader_persistent_workers: bool = True # 保持 worker 进程存活，减少创建开销 
    
    # --- 4. 数据与路径 ---
    dataset_name: str = "cifar10"        # 数据集名称 (60k images)
    dataset_cache_dir: str = "./datasets"                # 数据集下载/缓存目录
    output_dir: str = "output/cifar10-dit-64"            # 模型保存路径
    seed: int = 42
    
    # --- 5. 设备选择 (自动检测) ---
    @property
    def device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

config = TrainingConfig()
