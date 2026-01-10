# 第二章：DiT (Diffusion Transformer) 架构详解

本章节我们将深入了解文生图背后的核心引擎：**Diffusion Transformer (DiT)**。

## 1. 什么是 DiT？
传统的 Stable Diffusion (SD 1.5) 使用的是 **U-Net** 架构。而 DiT (如 SD 3.0, Sora) 则使用了 **Transformer** 架构。

Transformer 最初是为 NLP (自然语言处理) 设计的，那它是如何处理图片的呢？

### 1.1 Patchify (分块)
Transformer 处理的是序列 (Sequence)，如一句话中的单词序列。为了处理图片，我们需要把图片“切碎”。
*   假设输入图片是 `64x64`。
*   设定 `patch_size = 4`。
*   我们将图片切分为 `(64/4) * (64/4) = 16 * 16 = 256` 个小方块 (Patches)。
*   这 256 个 Patch 就相当于一句话里的 256 个单词。
*   DiT 将这些 Patch 展平并映射为向量，送入 Transformer。

### 1.2 Unpatchify (还原)
经过 Transformer 处理后，输出依然是 256 个向量。我们需要将它们重新拼回 `64x64` 的图片形状，这就是 Unpatchify。

## 2. 文生图的关键：Cross-Attention
如果只是把图片切块再拼回去，那只是一个普通的 Autoencoder。我们要实现的是 **Text-to-Image**，即让模型根据文本生成图片。

这是通过 **Cross-Attention (交叉注意力)** 机制实现的。

在 Transformer 的每一层中，都有两个注意力模块：
1.  **Self-Attention (自注意力)**: 图片 Patch 之间互相“交流”。例如，画“猫”的时候，耳朵的 Patch 需要知道眼睛在哪里。
2.  **Cross-Attention (交叉注意力)**: 图片 Patch 关注文本 Token。
    *   **Query (Q)**: 来自图片 Patch。
    *   **Key (K) / Value (V)**: 来自文本 (CLIP Text Encoder 的输出)。

形象地说，Cross-Attention 就像是图片在问文本：“这一块区域应该画什么颜色？是什么纹理？”文本回答：“这里是红色的眼睛。”

## 3. 我们的 TinyDiT 配置
为了在 Mac 上流畅运行，我们在 `src/config.py` 和 `src/model.py` 中定义了一个微型版 DiT：

| 参数 | 我们的 TinyDiT | 标准 DiT-XL/2 | 说明 |
| :--- | :--- | :--- | :--- |
| **Layers** | 6 | 28 | Transformer 的层数 (深度) |
| **Hidden Dim** | 256 | 1152 | 向量的宽度 |
| **Heads** | 4 | 16 | 并行处理的注意力头数 |
| **Params** | ~5 M | ~675 M | 我们的模型小了 100 倍以上 |

虽然模型很小，但它包含了完整 DiT 的所有核心组件，足以演示文生图的原理。

## 4. 文本编码器：CLIP
我们不从头训练文本理解能力，而是“借用” OpenAI 训练好的 CLIP 模型。
*   **CLIPTextModel**: 负责将文本 (如 "pikachu") 转换为高维向量 (Embeddings)。
*   我们在代码中设置 `text_encoder.requires_grad_(False)`，冻结它的参数。这意味着我们只训练 DiT 如何根据这些语义向量去生成图片，而不改变 CLIP 对文本的理解。
