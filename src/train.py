import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

try:
    from src.config import config
    from src.data_loader import get_dataloader
    from src.model import get_dit_model, get_text_encoder
except ImportError:
    from config import config
    from data_loader import get_dataloader
    from model import get_dit_model, get_text_encoder

def train():
    # 1. 初始化 Accelerator
    # Accelerator 会自动处理设备 (CPU/MPS/CUDA) 和混合精度
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=1, 
        log_with="all",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"🚀 开始训练! 输出目录: {config.output_dir}")
        print(f"💻 设备: {accelerator.device}")

    # 2. 准备组件
    # 噪声调度器 (Noise Scheduler): 负责加噪和去噪的数学计算
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 模型
    model = get_dit_model()
    
    # --- 自动检测 Resume (中断恢复) ---
    start_epoch = 0
    resume_path = None
    if os.path.exists(config.output_dir):
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-epoch-")]
        if checkpoints:
            # 按 epoch 数字排序
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = checkpoints[-1]
            resume_path = os.path.join(config.output_dir, latest_checkpoint)
            
            # 解析已完成的 Epoch
            start_epoch = int(latest_checkpoint.split("-")[-1])
            
            if start_epoch < config.num_epochs:
                print(f"🔄 检测到中断的训练: {latest_checkpoint}")
                print(f"📥 正在从 Epoch {start_epoch} 恢复权重...")
                # 加载权重覆盖原模型
                model = Transformer2DModel.from_pretrained(resume_path)
            else:
                print(f"✅ 检测到训练已完成 (Epoch {start_epoch}/{config.num_epochs})，若需重新训练请清理 output 目录。")
                start_epoch = 0 # 或者直接退出? 这里让它从 0 开始或者保持完成状态比较好。
                # 如果已经跑完了，就不加载了，或者加载了也没用，因为循环不会执行。
                # 让用户决定吧，这里假设用户想继续跑或者重跑。
                # 如果是 fully trained，range(5, 5) 是空的，直接结束。
                
    text_encoder = get_text_encoder()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 数据加载器
    train_dataloader = get_dataloader()
    
    # 学习率调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )

    # 3. 使用 Accelerator 包装对象
    # 注意：Text Encoder 不需要包装，因为它不参与训练 (冻结状态)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # 将 Text Encoder 移到正确的设备
    text_encoder.to(accelerator.device)

    # 4. 预计算 Text Embeddings (针对 CIFAR-10 等分类数据集的优化)
    # 如果是 CIFAR-10，只有 10 个固定的 Prompt，预先计算可以极大加速
    cached_text_embeddings = None
    if config.dataset_name == "cifar10":
        print("⚡️ 检测到 CIFAR-10 数据集，正在预计算 Text Embeddings 以加速训练...")
        cifar10_classes = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
        captions = [f"a photo of a {cifar10_classes[i]}" for i in range(10)]
        
        # 临时 Tokenizer (因为 dataloader 里的 tokenizer 不容易获取，这里重新加载一个也没事)
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            # 移动到设备
            input_ids = inputs.input_ids.to(accelerator.device)
            # [10, 77, 512]
            cached_text_embeddings = text_encoder(input_ids)[0]
        
        print(f"✅ Text Embeddings 预计算完成! Shape: {cached_text_embeddings.shape}")

    # 优化 3: 内存格式优化 (Channels Last)
    # 适用于卷积层较多的网络，在 GPU 上通常更快 (MPS 也有一定收益)
    model = model.to(memory_format=torch.channels_last)
    
    # 5. 训练循环
    global_step = 0
    
    # 如果是 Resume，需要快进 LR Scheduler 和 global_step
    if start_epoch > 0:
        steps_per_epoch = len(train_dataloader)
        resume_step = start_epoch * steps_per_epoch
        global_step = resume_step
        print(f"⏩ 正在快进 LR Scheduler 到 step {resume_step} ...")
        # 注意：这里简单的循环 step 可能比较慢，但最稳健
        # 对于 AdamW + Cosine，这一步很重要
        for _ in range(resume_step):
            lr_scheduler.step()
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # 优化: 确保输入也是 channels_last
            clean_images = batch["pixel_values"].to(memory_format=torch.channels_last)
            
            # --- A. 采样噪声 ---
            # 生成与输入图像形状一致的高斯噪声
            noise = torch.randn_like(clean_images)
            
            # --- B. 采样时间步 ---
            # 为每个样本随机选择一个时间步 t (0 到 999)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # --- C. 前向加噪 (Forward Diffusion) ---
            # 根据时间步 t，将噪声添加到图像上
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # --- D. 获取文本条件 ---
            # 性能优化：如果是 CIFAR-10 且有缓存，直接查表
            if cached_text_embeddings is not None and "labels" in batch:
                # batch["labels"] 是 [Batch] 的 tensor
                # 直接索引获取对应的 embeddings [Batch, 77, 512]
                encoder_hidden_states = cached_text_embeddings[batch["labels"]]
            else:
                # 常规流程：实时计算
                with torch.no_grad():
                    # CLIP Text Encoder 输出的 hidden_states
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # --- E. 模型预测 ---
            # DiT 预测噪声 (Predict the noise)
            # Hack: 传入 dummy class_labels 以满足 ada_norm_zero 的要求
            # 我们用全 0 作为 class label，相当于模型认为所有图片都属于同一个"类别"
            dummy_class_labels = torch.zeros(bsz, dtype=torch.long, device=clean_images.device)
            
            # 开启梯度累积上下文 (虽然这里是 1，但保持规范)
            with accelerator.accumulate(model):
                model_pred = model(
                    noisy_images, 
                    timestep=timesteps, 
                    encoder_hidden_states=encoder_hidden_states,
                    class_labels=dummy_class_labels
                ).sample

                # --- F. 计算 Loss ---
                # 目标是预测添加的那个噪声
                loss = F.mse_loss(model_pred, noise)

                # --- G. 反向传播 ---
                accelerator.backward(loss)
                
                # 梯度裁剪 (防止梯度爆炸)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True) # set_to_none=True 略微节省显存和操作

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

        # 每个 Epoch 结束后保存模型
        if accelerator.is_main_process:
            # 修改: 每个 Epoch 都保存，防止意外中断丢失进度
            save_path = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch+1}")
            # 保存 Unwrap 后的模型 (去除 DDP/MPS 包装)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path)
            print(f"\n💾 模型已保存至: {save_path}")

    print("🎉 训练完成！")

if __name__ == "__main__":
    train()
