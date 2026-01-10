import torch
import argparse
import os
from diffusers import Transformer2DModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.utils import save_image
from tqdm.auto import tqdm

try:
    from src.config import config
except ImportError:
    from config import config

def inference(prompt, model_path, guidance_scale=7.5, num_steps=50):
    """
    文生图推理脚本
    
    Args:
        prompt: 文本提示词
        model_path: 模型权重路径
        guidance_scale: CFG (Classifier-Free Guidance) 强度，通常 7.5 效果较好
        num_steps: 采样步数
    """
    # 1. 设备配置
    device = torch.device(config.device)
    print(f"🚀 使用设备: {device}")
    print(f"🎨 提示词: '{prompt}'")

    # 2. 加载模型
    print(f"📥 加载模型: {model_path}")

    # 加载我们训练好的 DiT 模型
    # 注意：这里我们使用 Transformer2DModel.from_pretrained 加载
    # 并且必须确保配置与训练时一致 (model.py 中的 get_dit_model)
    
    # 尝试直接加载
    # 注意：如果 Transformer2DModel 实例化时没有正确识别为支持 cross-attention，
    # 它可能不会接受 encoder_hidden_states。
    # 在 diffusers 中，DiT 通常通过 class labels 控制，而 Cross-Attention 需要特定的配置。
    # 为了保险，我们强制使用 get_dit_model() 构建模型，然后加载权重。
    
    try:
        # 强制使用代码中定义的结构，确保配置正确 (包含 cross_attention_dim)
        from model import get_dit_model
        print("🏗️ 使用 model.py 定义构建模型结构...")
        model = get_dit_model()
        
        # 加载权重
        from diffusers.models.modeling_utils import load_state_dict
        if os.path.isdir(model_path):
             # 尝试查找 safetensors
             weight_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
             if not os.path.exists(weight_path):
                 weight_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
        else:
             weight_path = model_path
             
        print(f"⚖️ 加载权重: {weight_path}")
        state_dict = load_state_dict(weight_path)
        
        # 过滤掉不匹配的键 (如果有的话，例如 unexpected keys)
        # model.load_state_dict(state_dict, strict=False) 
        # 使用 strict=True 以确保我们训练的权重完全匹配
        # 如果报错，说明保存的权重和当前模型定义不一致
        model.load_state_dict(state_dict)
        
    except Exception as e:
        print(f"⚠️ 自定义加载失败: {e}")
        print("尝试回退到 from_pretrained...")
        model = Transformer2DModel.from_pretrained(model_path, use_safetensors=True)

    model.to(device)
    model.eval()
    
    # 加载 CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # 3. 准备文本 Embeddings (含 CFG 处理)
    # CFG 需要两个输入：有提示的 (Conditional) 和 空提示的 (Unconditional)
    
    # (a) Conditional Embeddings
    text_input = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    # (b) Unconditional Embeddings (空文本)
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # (c) 拼接 (Batch Size = 2)
    # 为了并行计算，我们将它们拼在一起
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    # 4. 初始化噪声
    # 从纯高斯噪声开始
    generator = torch.Generator(device=device).manual_seed(config.seed)
    latents = torch.randn(
        (1, 3, config.image_size, config.image_size),
        generator=generator,
        device=device
    )
    
    # 5. 设置 Scheduler
    # 推理时我们可以使用更快的 Scheduler，这里为了简单仍用 DDPM
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_steps)

    # 6. 采样循环 (Denoising Loop)
    print("✨ 开始生成...")
    for t in tqdm(scheduler.timesteps):
        # 1. 扩展 Latents 以适应 CFG (Batch Size * 2)
        latent_model_input = torch.cat([latents] * 2)
        
        # 2. 模型预测噪声
        # 同样需要构造 dummy class labels (2*BatchSize)
        dummy_class_labels = torch.zeros(latent_model_input.shape[0], dtype=torch.long, device=device)
        
        # 确保 timestep 是 1D tensor
        # t 只是一个标量 (int 或 float)，我们需要把它扩展成 (batch_size,)
        timestep_tensor = torch.tensor([t] * latent_model_input.shape[0], device=device)
        
        with torch.no_grad():
            noise_pred = model(
                latent_model_input, 
                timestep=timestep_tensor, 
                encoder_hidden_states=text_embeddings,
                class_labels=dummy_class_labels
            ).sample

        # 3. 应用 CFG (Classifier-Free Guidance)
        # noise_pred 包含了 [uncond_pred, cond_pred]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # 核心公式: result = uncond + scale * (cond - uncond)
        # scale > 1 时，会强化文本对生成结果的影响
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 4. 计算前一步的 Latents (去噪)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 7. 后处理与保存
    # [-1, 1] -> [0, 1]
    image = (latents / 2 + 0.5).clamp(0, 1)
    image = image.cpu()
    
    output_filename = f"generated_{prompt.replace(' ', '_')}.png"
    save_image(image, output_filename)
    print(f"✅ 图片已保存至: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a red pokemon", help="生成的提示词")
    
    # 自动查找最新的 checkpoint
    default_path = None
    if os.path.exists(config.output_dir):
        # 列出所有 checkpoint 文件夹
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-epoch-")]
        if checkpoints:
            # 排序规则：提取 epoch 数字进行排序 (checkpoint-epoch-1, checkpoint-epoch-2, ...)
            # 假设文件夹格式严格为 checkpoint-epoch-N
            try:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                default_path = os.path.join(config.output_dir, checkpoints[-1])
            except ValueError:
                # 如果格式不对，就按字母序
                checkpoints.sort()
                default_path = os.path.join(config.output_dir, checkpoints[-1])
    
    # 如果没找到，回退到默认的 checkpoint-epoch-50 (用于提示用户)
    if default_path is None:
        default_path = os.path.join(config.output_dir, "checkpoint-epoch-50")

    parser.add_argument("--model_path", type=str, default=default_path)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
         print(f"⚠️ 警告: 模型路径 {args.model_path} 不存在。请先训练模型或检查路径。")
         print(f"提示: 您可以使用 --model_path 指定具体路径。")
    else:
        inference(args.prompt, args.model_path)
