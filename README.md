# Mac 文生图 (Text-to-Image) 实战教程

欢迎来到这个专为 **Mac (Apple Silicon)** 用户设计的文生图开发教程！

本项目旨在带你从零开始，使用 Hugging Face `diffusers` 库，构建并训练一个基于 **DiT (Diffusion Transformer)** 的微型文生图模型。

## 🎯 项目亮点
*   **Mac 友好**: 针对 M1/M2/M3 芯片优化，使用 MPS (Metal Performance Shaders) 加速。
*   **极简架构**: 采用 Pixel-Space DiT (64x64)，无需昂贵的显卡即可在几小时内完成训练。
*   **代码即文档**: 所有源码均包含详尽的中文注释。
*   **理论结合**: 配套 Markdown 教程，深入浅出讲解 Diffusion 和 Transformer 原理。

## 📚 教程目录
我们建议按照以下顺序阅读文档并运行代码：

1.  **[环境与数据准备](docs/01_environment_and_data.md)**
    *   了解 Mac MPS 加速。
    *   运行 `python src/data_loader.py` 下载并预览 Pokemon 数据集。
2.  **[DiT 模型架构](docs/02_dit_architecture.md)**
    *   图解 Patchify 和 Transformer。
    *   运行 `python src/model.py` 查看模型结构。
3.  **[训练动态详解](docs/03_training_dynamics.md)**
    *   理解扩散模型的加噪与去噪过程。
    *   运行 `python src/train.py` 开始预训练！
4.  **[推理与技巧](docs/04_inference_and_tricks.md)**
    *   学习 Classifier-Free Guidance (CFG)。
    *   运行 `python src/inference.py` 生成你自己的 Pokemon。

## 🚀 快速开始

### 1. 安装依赖
```bash
cd t2i_tutorial
pip install -r requirements.txt
```

### 2. 数据预览
确保网络通畅 (需访问 Hugging Face)，运行：
```bash
python src/data_loader.py
```
成功后会生成 `sample_pokemon.png`。

### 3. 开始训练 (Pre-training)
```bash
python src/train.py
```
*   **耗时**: 在 M1 Pro 上，训练 50 个 Epoch 约需 1-2 小时。
*   **输出**: 模型将保存在 `output/pokemon-dit-64`。

### 4. 文生图推理
训练完成后，尝试生成：
```bash
python src/inference.py --prompt "a blue dragon with fire"
```
生成的图片将保存在当前目录。

### 5. 微调 (SFT)
如果你有自己的小数据集，或者想尝试微调：
```bash
python src/finetune.py --model_path output/pokemon-dit-64/checkpoint-epoch-50
```

## 🛠️ 技术栈
*   **Framework**: PyTorch (MPS support)
*   **Library**: Diffusers, Transformers, Accelerate
*   **Dataset**: LambdaLabs Pokemon BLIP Captions

---
*Happy Coding on Mac! 🍎*
