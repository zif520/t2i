# 文生图 DiT 教程开发计划 (增强版：代码+文档并行)

根据您的要求，我们将采用“代码即教程”的模式，确保每一行代码都有详尽的注释，并同步产出 Markdown 教程文档。

## 核心策略：代码与文档同步
我们将按照“概念讲解 -> 代码实现 -> 运行验证”的循环进行开发。

### 模块 1: 环境与数据基石
*   **Docs**: `docs/01_environment_and_data.md` - 解释 Mac MPS 加速原理，介绍 Hugging Face `datasets` 库。
*   **Code**: `src/data_loader.py`
    *   **功能**: 自动下载 Pokemon 图文数据集，执行 Resize (64x64) 和 Normalize。
    *   **注释重点**: 解释为何需要归一化到 `[-1, 1]`，以及 `DataLoader` 的 `num_workers` 在 Mac 上的设置技巧。

### 模块 2: 深入 DiT 模型架构
*   **Docs**: `docs/02_dit_architecture.md` - 图解 Diffusion Transformer (DiT)。解释 Patchify、Positional Embedding 和 Cross-Attention 如何融合文本信息。
*   **Code**: `src/model.py`
    *   **功能**: 配置 `Transformer2DModel`。
    *   **注释重点**: 详细说明 `patch_size` 对计算量的影响，以及 `cross_attention_dim` 如何对接文本特征。

### 模块 3: 训练循环与核心原理
*   **Docs**: `docs/03_training_dynamics.md` - 详解扩散过程 (Forward Process) 和去噪过程 (Reverse Process)。解释 MSE Loss 在预测噪声时的物理含义。
*   **Code**: `src/train.py` (预训练) & `src/finetune.py` (SFT)
    *   **功能**: 完整的训练 Loop，支持断点续训。
    *   **注释重点**: 逐行解释 `noise_scheduler.add_noise` 和 `loss.backward()`，以及 `accelerate` 库如何管理设备。

### 模块 4: 推理与生成
*   **Docs**: `docs/04_inference_and_tricks.md` - 介绍采样器 (DDPM vs DDIM) 的区别，以及 Classifier-Free Guidance (CFG) 如何提升文本依从性。
*   **Code**: `src/inference.py`
    *   **功能**: 加载训练好的模型，输入文本生成图片。

## 项目交付物
```text
t2i_tutorial/
├── docs/                   # 配套教程文档 (Markdown)
├── src/                    # 源代码 (含详细教学注释)
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── requirements.txt
└── README.md               # 教程入口索引
```

我们将从**模块 1**开始，逐步构建这个系统。每完成一个模块，都会提交代码和对应的文档。