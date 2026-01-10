# 第四章：推理与魔法 (Inference & Tricks)

本章节介绍如何使用训练好的模型生成图片，并揭示让文生图效果惊艳的秘密武器：**Classifier-Free Guidance (CFG)**。

## 1. 采样过程 (Sampling)
训练时，我们是一步到位地预测噪声。
推理时，我们需要**一步步地**把噪声减掉。

1.  **初始状态**: 一张完全由随机数字组成的图 (纯噪声)。
2.  **Step 1**: 问模型“如果是第 1000 步，这上面的噪声大概是多少？”模型回答后，我们减去一小部分噪声。
3.  **Step 2**: 拿着减去一点噪声的图，问模型“如果是第 980 步，噪声是多少？”...
4.  **Loop**: 重复几十次，直到噪声被清理干净，露出底下的图片。

## 2. Classifier-Free Guidance (CFG)
如果你直接运行模型，你会发现生成的图片虽然像 Pokemon，但可能不太听你的话 (比如你让它画红色的，它画了蓝色的)。

这是因为模型在训练时，既要看图片，又要看文本，它可能会“偷懒”，忽略文本，只根据图片本身的规律去生成。

**CFG** 是一种强迫模型听话的技术：
1.  **无条件预测 (Unconditional)**: 我们给模型输入一个**空文本 ""**，让它自由发挥。
    *   `pred_uncond` = 模型觉得“大概是个 Pokemon 就行”。
2.  **有条件预测 (Conditional)**: 我们给模型输入**你的提示词 "red pokemon"**。
    *   `pred_text` = 模型觉得“这是一个红色的 Pokemon”。
3.  **做差与放大**:
    *   `diff = pred_text - pred_uncond`。这个差值代表了**“红色”这个概念带来的独特变化**。
    *   `final = pred_uncond + scale * diff`。
    *   我们把这个“独特变化”放大 (例如放大 7.5 倍)。

这就像是对模型说：“把你觉得是红色的那些特征，给我狠狠地加强！”

## 3. 代码实现
在 `src/inference.py` 中：
```python
# 拼接输入，一次性计算
latent_model_input = torch.cat([latents] * 2)
text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

# 模型预测
noise_pred = model(..., encoder_hidden_states=text_embeddings).sample

# 拆分结果
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

# CFG 公式
noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
```
这个简单的公式是 DALL-E 2, Stable Diffusion 等现代模型成功的基石。
