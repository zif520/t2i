# 第三章：训练动态 (Training Dynamics)

本章节将解析 `src/train.py` 中的核心逻辑：模型是如何“学会”画画的。

## 1. 扩散模型的本质
扩散模型 (Diffusion Model) 的训练过程其实非常反直觉。它不是直接学习“如何画一只皮卡丘”，而是学习“如何把一张满是噪点的图修好”。

训练过程分为两步：
1.  **加噪 (Forward)**: 我们拿一张清晰的皮卡丘图片，往上面撒一点噪声，再撒一点...直到它完全变成一张雪花屏 (高斯噪声)。
2.  **去噪 (Reverse)**: 模型的任务是，给它一张加了噪的图，让它预测**刚才加了什么噪声**。如果模型能准确预测出噪声，我们把这个噪声减掉，图就变清晰了一点点。

## 2. 训练循环详解
在 `src/train.py` 中，每一个 Batch 都发生了以下故事：

### 2.1 准备食材 (Input)
```python
clean_images = batch["pixel_values"]
encoder_hidden_states = text_encoder(batch["input_ids"])[0]
```
我们有清晰的图片，还有这张图片对应的文本描述 (经过 CLIP 编码)。

### 2.2 随机破坏 (Add Noise)
```python
noise = torch.randn_like(clean_images)
timesteps = torch.randint(...)
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
```
*   我们随机选一个时间点 `t` (比如第 500 步)。
*   `noise_scheduler` 会计算出第 500 步时，图片应该有多模糊，并把噪声加进去。
*   现在我们有了 `noisy_images` (模糊的图)。

### 2.3 模型的挑战 (Prediction)
```python
model_pred = model(noisy_images, timestep=timesteps, encoder_hidden_states=encoder_hidden_states).sample
```
我们把 **模糊的图**、**时间步 t** 和 **文本提示** 扔给 DiT 模型。
模型的任务是：**请告诉我，刚才加的那个噪声长什么样？**

注意：模型通过 Cross-Attention 读取文本提示，以此来辅助判断。例如，如果文本是 "Pikachu"，模型就会知道这个模糊的一团黄影子里应该包含皮卡丘的特征，从而推断出噪声的形状。

### 2.4 计算误差 (Loss)
```python
loss = F.mse_loss(model_pred, noise)
```
*   `noise`: 真实的噪声 (正确答案)。
*   `model_pred`: 模型猜的噪声。
*   `MSE Loss`: 计算两者差距的平方。差距越小，说明模型猜得越准。

### 2.5 修正模型 (Backpropagation)
通过 `loss.backward()`，我们告诉模型参数该往哪个方向调整，以便下次猜得更准。

## 3. Accelerator 的魔力
我们在代码中使用了 Hugging Face 的 `Accelerate` 库：
```python
accelerator = Accelerator(...)
model, optimizer, ... = accelerator.prepare(...)
```
这几行代码让我们可以无缝切换硬件。
*   在 Mac 上，它自动使用 MPS。
*   在 NVIDIA 机器上，它自动使用 CUDA。
*   它还负责处理 `backward()` 和梯度更新，让代码更简洁。
