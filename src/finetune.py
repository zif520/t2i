import os
import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, Transformer2DModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

try:
    from src.config import config
    from src.data_loader import get_dataloader
    from src.model import get_text_encoder
except ImportError:
    from config import config
    from data_loader import get_dataloader
    from model import get_text_encoder

def finetune(pretrained_model_path):
    """
    SFT (Supervised Fine-Tuning) 微调脚本
    
    演示如何加载预训练好的 DiT 模型，并在小数据集上继续训练。
    """
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=1, 
        log_with="all",
        project_dir=os.path.join(config.output_dir, "finetune_logs")
    )
    
    if accelerator.is_main_process:
        print(f"🚀 开始 SFT 微调! 基础模型: {pretrained_model_path}")

    # 2. 加载预训练模型
    # 关键点：我们不是从头初始化，而是加载训练好的权重
    print(f"📥 正在加载权重: {pretrained_model_path}")
    model = Transformer2DModel.from_pretrained(pretrained_model_path)
    
    text_encoder = get_text_encoder()
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 3. 微调设置
    # SFT 通常使用更低的学习率
    finetune_lr = 1e-5 
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
    
    # 这里为了演示，我们依然使用 Pokemon 数据集
    # 在实际应用中，你可以替换为你的垂直领域数据集 (如 "中国山水画" 数据集)
    train_dataloader = get_dataloader()
    
    lr_scheduler = get_scheduler(
        "constant", # 微调通常使用常数学习率
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )

    # 4. Prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    text_encoder.to(accelerator.device)

    # 5. 训练循环 (简化版)
    # 这里的逻辑与 train.py 完全一致
    model.train()
    for epoch in range(5): # 微调通常只需要很少的 Epoch
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Finetune Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["pixel_values"]
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            
            # 同样添加 dummy class labels
            dummy_class_labels = torch.zeros(clean_images.shape[0], dtype=torch.long, device=clean_images.device)
            
            model_pred = model(
                noisy_images, 
                timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states,
                class_labels=dummy_class_labels
            ).sample
            
            loss = F.mse_loss(model_pred, noise)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

    # 保存微调后的模型
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, "finetuned-dit")
        accelerator.unwrap_model(model).save_pretrained(save_path)
        print(f"✅ 微调完成，模型已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认加载 output/pokemon-dit-64 下的最新 checkpoint (假设用户已经跑了 train.py)
    # 如果没有，用户需要手动指定
    default_path = os.path.join(config.output_dir, "checkpoint-epoch-50")
    parser.add_argument("--model_path", type=str, default=default_path, help="预训练模型的路径")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"⚠️ 警告: 路径 {args.model_path} 不存在。请先运行 src/train.py 或指定正确的路径。")
    else:
        finetune(args.model_path)
