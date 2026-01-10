import torch
from diffusers import Transformer2DModel
from transformers import CLIPTextModel
try:
    from src.config import config
except ImportError:
    from config import config

def get_dit_model():
    """
    构建 DiT (Diffusion Transformer) 模型
    
    使用 diffusers 库中的 Transformer2DModel。
    这相当于一个“图像版的 BERT”：
    1. 输入图片被切分为 patches (类似单词)。
    2. 经过多层 Transformer 处理。
    3. 输出同样大小的噪声预测。
    """
    print(f"🏗️ 正在构建 DiT 模型 (Patch Size={config.patch_size})...")
    
    model = Transformer2DModel(
        sample_size=config.image_size,      # 输入大小 64x64
        patch_size=config.patch_size,       # Patch 大小 4x4 -> 序列长度 = (64/4)^2 = 256
        in_channels=3,                      # 输入通道 RGB
        num_layers=config.num_layers,       # Transformer 层数
        attention_head_dim=config.attention_head_dim,
        num_attention_heads=config.num_attention_heads,
        out_channels=3,                     # 输出通道 (预测噪声，与输入形状一致)
        
        # --- 关键：条件生成配置 ---
        cross_attention_dim=512,            # 文本特征的维度 (CLIP-ViT-Base-Patch32 的输出维度是 512)
        # 这允许模型在每一层 Transformer 中通过 Cross-Attention "关注" 文本描述
        
        # DiT 必备参数
        norm_type="ada_norm_zero",  # 当使用 patch_size 时，通常使用 AdaLayerNormZero
        num_embeds_ada_norm=1000,   # 必须匹配 DDPMScheduler 的 num_train_timesteps
        
        # 关键修复：我们不需要类别条件 (class_labels)，因为是 Text-to-Image
        # AdaLayerNormZero 默认需要 class_labels，这里我们需要让它知道我们不传 class_labels
        # 但 diffusers 的实现中，ada_norm_zero 强绑定了 class embedding。
        # 实际上，对于纯文生图 DiT，我们应该使用 norm_type="layer_norm" 但 diffusers 限制了 patch_size 必须配 ada_norm。
        # 替代方案：构造一个 dummy class label 或者使用不同的 norm 策略。
        # 更简单的方案：不使用 patch_size (即不使用 DiT)，或者手动构造 class_labels。
        # 
        # 为了让 DiT 跑通，我们这里传入一个假的 class label (全0) 在 train.py 中，
        # 并在这里设置 class_embed_type="timestep" (但这不被 Transformer2DModel 支持)
        # 
        # 正确姿势：diffusers 的 Transformer2DModel 在 patch_size 模式下主要设计给 Class-Conditioned 生成 (如 DiT paper)。
        # 对于 Text-Conditioned，通常不使用 ada_norm_zero 或者需要 hack。
        # 
        # Hack: 我们在 train.py 中传入 class_labels=None，但这里必须去掉对 class_labels 的依赖。
        # 然而 Transformer2DModel 源码强制检查 num_embeds_ada_norm。
        # 
        # 让我们尝试改为 norm_type="layer_norm" 并去掉 patch_size (退化为普通 Transformer)，
        # 或者保留 patch_size 但在 train.py 中传入 dummy class_labels。
        # 
        # 决定：为了保持 DiT 特性，我们在 train.py 中传入 dummy class_labels。
    )
    return model

def get_text_encoder():
    """
    加载预训练的文本编码器 (CLIP)
    """
    print(f"🧠 正在加载 Text Encoder: openai/clip-vit-base-patch32 ...")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # 冻结 Text Encoder 参数
    # 我们只训练 DiT，不训练 CLIP，这样可以节省大量显存和计算资源
    text_encoder.requires_grad_(False)
    
    return text_encoder

if __name__ == "__main__":
    # 测试模型构建
    print("🧪 测试模型架构...")
    
    dit = get_dit_model()
    text_encoder = get_text_encoder()
    
    # 统计参数量
    dit_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    print(f"📦 DiT Trainable Params: {dit_params / 1e6:.2f} M (Million)")
    
    # 模拟一次前向传播 (Forward Pass)
    # 1. 模拟图像输入 (Batch=2, Channel=3, H=64, W=64)
    dummy_image = torch.randn(2, 3, config.image_size, config.image_size)
    # 2. 模拟时间步 (Timestep)
    dummy_timestep = torch.tensor([0, 100])
    # 3. 模拟文本特征 (Batch=2, Seq=77, Dim=512)
    dummy_encoder_hidden_states = torch.randn(2, 77, 512)
    
    # 4. 模型预测
    # 输出应该是 [2, 3, 64, 64]
    output = dit(
        dummy_image, 
        timestep=dummy_timestep, 
        encoder_hidden_states=dummy_encoder_hidden_states
    ).sample
    
    print(f"✅ Forward Pass 成功!")
    print(f"Input Shape: {dummy_image.shape}")
    print(f"Output Shape: {output.shape}")
    
    if output.shape == dummy_image.shape:
        print("🎉 维度匹配，架构验证通过。")
