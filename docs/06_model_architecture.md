# DiT 模型结构详解 (Model Architecture)

本文档通过图表详细解析本项目中使用的 **TinyDiT** 架构。它基于 `diffusers.Transformer2DModel` 构建，并针对文生图任务进行了适配。

## 1. 核心架构图 (Mermaid Diagram)

```mermaid
graph TD
    %% --- 输入层 ---
    subgraph Inputs [模型输入]
        ImgInput[Noisy Image<br>Batch, 3, 64, 64]
        TimeInput[Timestep t<br>Batch]
        TextInput[Text Embeddings<br>Batch, 77, 512]
        DummyClass[Dummy Class Labels<br>Batch]
    end

    %% --- Patchify 阶段 ---
    subgraph PatchProcessing [图像分块 Patchify]
        ImgInput --> Conv2D[Conv2d Embedding<br>Kernel=4, Stride=4]
        Conv2D --> Flatten[Flatten -> Batch, 256, 256]
        Flatten --> PosEmbed[Add Positional Embeddings]
        PosEmbed --> Tokens[Image Tokens]
    end

    %% --- 条件处理 ---
    subgraph Conditioning [条件注入]
        TimeInput & DummyClass --> TimeProj[Timestep/Class Projection]
        TimeProj --> AdaNormSignal[AdaLayerNormZero 信号<br>Scale, Shift, Gate]
    end

    %% --- Transformer 主体 ---
    subgraph TransformerBody [Transformer Blocks x6 Layers]
        direction TB
        Tokens --> Block1[Block 1]
        Block1 --> Block2[Block 2]
        Block2 --> Block3[...]
        Block3 --> Block6[Block 6]
        
        %% 内部结构放大
        subgraph BlockDetail [Transformer Block 内部细节]
            BlockIn[Input Tokens] --> AdaNorm1[AdaLayerNormZero 1<br>自适应归一化]
            AdaNormSignal -.-> AdaNorm1
            
            AdaNorm1 --> SelfAttn[Self-Attention<br>图像内部关联]
            SelfAttn --> Resid1[Residual Connection]
            
            Resid1 --> AdaNorm2[AdaLayerNormZero 2]
            AdaNormSignal -.-> AdaNorm2
            
            AdaNorm2 --> CrossAttn[Cross-Attention<br>图像-文本关联]
            TextInput -.-> CrossAttn
            CrossAttn --> Resid2[Residual Connection]
            
            Resid2 --> FFN[Feed Forward Network]
            FFN --> BlockOut[Output Tokens]
        end
    end

    %% --- 输出层 ---
    subgraph OutputLayer [Unpatchify & 输出]
        Block6 --> FinalNorm[Final Norm AdaLayerNormZero]
        AdaNormSignal -.-> FinalNorm
        FinalNorm --> LinearProj[Linear Projection]
        LinearProj --> Reshape[Reshape -> Batch, 3, 64, 64]
        Reshape --> PredNoise[Predicted Noise]
    end

    Inputs --> PatchProcessing
    PatchProcessing --> TransformerBody
    TransformerBody --> OutputLayer
```

## 2. 关键组件解析

### A. Patchify (分块嵌入)
*   **作用**: 将二维图片转换为 Transformer 可以处理的一维序列。
*   **实现**: 使用一个卷积层 (`Conv2d`)，核大小和步长都等于 `patch_size` (4)。
*   **维度变化**: `[B, 3, 64, 64]` -> `[B, 256, 16, 16]` -> Flatten -> `[B, 256, 256]` (序列长度=256, 维度=256)。

### B. AdaLayerNormZero (自适应归一化)
*   **作用**: 这是 DiT 的核心机制。它不使用标准的 LayerNorm，而是根据 **时间步 (t)** 和 **类别 (Class)** 动态生成归一化的参数 (Scale & Shift)。
*   **意义**: 这使得模型知道“现在是第几步扩散过程”，从而控制生成的进度。
*   **Hack**: 由于我们是做文生图，但在使用 `patch_size` 时 `diffusers` 强制要求这种归一化方式，所以我们传入了全 0 的 `dummy_class_labels` 来满足 API 要求，主要依赖时间步信息。

### C. Cross-Attention (交叉注意力)
*   **作用**: 将文本信息注入到图像生成过程中。
*   **机制**:
    *   **Query**: 来自图像的 Tokens。
    *   **Key/Value**: 来自 CLIP 编码的文本 Embeddings (77个 Token)。
*   **结果**: 图像的每个 Patch 都能“看到”文本描述，从而生成符合描述的内容（例如“红色”的文本特征会引导“眼睛”区域的 Patch 变成红色）。
