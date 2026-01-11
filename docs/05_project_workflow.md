# 项目全流程图解 (Project Workflow)

本文档详细梳理了 `TinyDiT` 项目从数据准备、模型初始化、训练循环到最终推理的完整工作流。

## 1. 整体架构概览 (Mermaid Flowchart)

```mermaid
graph TD
    %% --- 初始化阶段 ---
    subgraph Init [初始化阶段]
        direction TB
        Config[src/config.py<br>配置加载] --> EnvCheck[设备检测 Mac MPS]
        EnvCheck --> LoadData[加载数据 src/data_loader.py]
        
        subgraph DataPipeline [数据预处理流水线]
            Raw[CIFAR-10 Raw] --> Transform[Resize/Normalize]
            Transform --> Label2Text[Label转Text<br>airplane -> a photo of a airplane]
            Label2Text --> Cache[In-Memory Cache<br>全量预加载至内存]
        end
        LoadData --> DataPipeline
        
        EnvCheck --> BuildModel[构建模型 src/model.py]
        BuildModel --> DiT[DiT Transformer2DModel]
        BuildModel --> CLIP[CLIP Text Encoder Frozen]
        
        BuildModel --> ResumeCheck{检查中断恢复?}
        ResumeCheck -- Yes --> LoadWeights[加载最新 Checkpoint]
        ResumeCheck -- No --> RandomInit[随机初始化]
    end

    %% --- 训练阶段 ---
    subgraph Train [训练循环 src/train.py]
        direction TB
        Cache --> Batch[获取 Batch Images, Labels]
        
        %% 文本条件分支
        Batch --> TextLogic{Text Embedding 策略}
        TextLogic -- Cache Hit --> CachedEmb[使用缓存 Embedding]
        TextLogic -- Cache Miss --> RunCLIP[运行 CLIP Encoder]
        
        %% 图像加噪分支
        Batch --> SampleNoise[采样高斯噪声]
        Batch --> SampleTime[采样时间步 t]
        SampleNoise & Batch & SampleTime --> AddNoise[前向加噪<br>Forward Diffusion]
        
        %% 模型计算
        AddNoise & SampleTime & CachedEmb --> ModelForward[DiT 前向传播]
        ModelForward --> PredNoise[预测噪声]
        
        %% 优化
        PredNoise & SampleNoise --> CalcLoss[计算 MSE Loss]
        CalcLoss --> Backprop[反向传播 & 梯度裁剪]
        Backprop --> OptimStep[Optimizer Update]
        
        OptimStep --> Save{End of Epoch?}
        Save -- Yes --> Checkpoint[保存模型<br>output/checkpoint-epoch-X]
    end

    %% --- 推理阶段 ---
    subgraph Infer [推理/生成 src/inference.py]
        UserPrompt[用户提示词] --> InfCLIP[CLIP Encode]
        RandomLatents[随机初始噪声] --> DDIM[调度器去噪循环]
        
        DDIM -- t=999...0 --> InfModel[DiT 预测噪声]
        InfModel --> DenoiseStep[去除一步噪声]
        DenoiseStep --> DDIM
        
        DDIM -- 完成 --> GenImage[生成图像]
    end

    Init --> Train
    Train --> Infer
```

## 2. 关键流程详解

### A. 数据加速 (Data Acceleration)
为了适配 Mac 的统一内存架构，我们实施了激进的数据优化：
1.  **In-Memory Caching**: 启动时将 60,000 张 CIFAR-10 图片全部解码、处理并存储为 Tensor 放在内存中 (约 700MB)。这消除了训练时的磁盘 IO 和 CPU 预处理开销。
2.  **Text Embedding Caching**: 对于固定类别的 CIFAR-10，我们只计算一次 CLIP Embedding 并缓存。训练时直接查表，不再运行庞大的 CLIP 模型。

### B. 混合精度训练 (Mixed Precision)
虽然配置中支持 `fp16`，但在 Mac MPS 上，为了稳定性我们目前默认使用 `fp32` (或者由 Accelerator 自动管理)。代码中保留了 `accelerator.accumulate` 接口，支持梯度累积。

### C. 断点续传 (Resume)
`src/train.py` 包含自动检测逻辑：
*   检查 `output_dir` 下的 checkpoints。
*   加载最新的权重。
*   自动“快进” DataLoader 和 Scheduler 到对应的步数，确保训练无缝继续。
