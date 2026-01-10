# 第一章：环境搭建与数据准备

本章节将指导你配置 Mac 上的深度学习开发环境，并准备用于训练文生图模型的 Pokemon 数据集。

## 1. 为什么选择 Mac + MPS ?
在过去，深度学习训练主要依赖 NVIDIA GPU (CUDA)。但随着 Apple Silicon (M1/M2/M3) 的推出，Apple 提供了 **MPS (Metal Performance Shaders)** 后端，允许 PyTorch 直接利用 Mac 的 GPU 进行加速。

虽然 Mac 的显存 (统一内存) 和算力不如高端 NVIDIA 显卡，但对于**微调 (Fine-tuning)** 或 **小型模型 (Tiny Models)** 的训练来说，它是完全够用的，且无需昂贵的云服务器。

## 2. 依赖库解析
我们在 `requirements.txt` 中列出了核心依赖：

*   **`diffusers`**: Hugging Face 推出的扩散模型库。它不仅包含 Stable Diffusion，还提供了构建块 (Building Blocks) 如 `Transformer2DModel` (DiT的核心)，让我们能像搭积木一样构建自己的模型。
*   **`accelerate`**: 这是一个“让 PyTorch 代码跑在任何设备上”的库。它会自动检测你是用 CPU、CUDA 还是 MPS，并处理多卡通信。对于 Mac 用户，它能自动处理 MPS 的设备放置。
*   **`datasets`**: 一行代码下载和管理海量数据集。

## 3. 数据集：Pokemon BLIP Captions
为了演示“文生图”，我们需要 (Image, Text) 对。
我们选用 `lambdalabs/pokemon-blip-captions` 数据集。
*   **Image**: Pokemon 的图片。
*   **Text**: 使用 BLIP 模型自动生成的描述 (Caption)，例如 "a drawing of a green pokemon with red eyes"。

### 3.1 图像预处理 (Preprocessing)
在 `src/data_loader.py` 中，我们定义了以下变换：
1.  **Resize (64x64)**: 原始图片大小不一，我们需要统一尺寸。64x64 虽然像素低，但训练速度极快，适合教程演示。
2.  **Normalize ([-1, 1])**: 
    *   图片通常是 `[0, 255]` 的整数。
    *   `ToTensor()` 将其转为 `[0, 1]` 的浮点数。
    *   `Normalize([0.5], [0.5])` 执行 `(x - 0.5) / 0.5`，将其映射到 `[-1, 1]`。
    *   **原因**: 扩散模型预测的是“噪声”，而初始的高斯噪声分布在 0 附近 (正负都有)。将输入图片也归一化到这个范围，有助于模型训练的稳定性。

### 3.2 文本处理 (Tokenization)
计算机看不懂文本，只能看懂数字。我们需要将文本转换为 Token ID 序列。
*   我们使用 **CLIP Tokenizer**。
*   CLIP 是一个连接文本和图像的预训练模型。我们使用它的 Tokenizer，是为了稍后能使用它的 Text Encoder 提取语义特征。
*   输出形状通常是 `[Batch_Size, 77]`，其中 77 是 CLIP 设定的最大序列长度。

## 4. 动手实践
运行以下命令测试数据加载器：
```bash
python src/data_loader.py
```
如果成功，你应该能看到 Batch Shape 的打印信息，并在目录下生成一张 `sample_pokemon.png`。
