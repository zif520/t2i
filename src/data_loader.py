import torch
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader
# 引用本地配置
try:
    from src.config import config
except ImportError:
    from config import config

class TextToImageDataset:
    """
    通用文生图数据集加载器
    
    支持:
    1. 图像-文本对数据集 (如 Pokemon)
    2. 图像-标签数据集 (如 CIFAR-10)，会自动将标签转换为文本提示
    """
    def __init__(self):
        print(f"📚 正在加载 Tokenizer: openai/clip-vit-base-patch32 ...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 图像增强与预处理
        self.transforms = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # CIFAR-10 类别映射
        self.cifar10_classes = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
        
        # 内部状态，用于 transform
        self.image_col = "image"
        self.text_col = "text"
        self.label_col = None

    def _transform_function(self, examples):
        """
        数据转换函数 (必须是 picklable 的，不能是局部函数)
        """
        # 1. 处理图像
        pixel_values = [self.transforms(img.convert("RGB")) for img in examples[self.image_col]]
        
        # 2. 处理文本
        captions = []
        if self.text_col and self.text_col in examples:
            captions = examples[self.text_col]
        elif self.label_col and self.label_col in examples:
            # 如果是分类数据集，根据 Label 生成 Prompt
            labels = examples[self.label_col]
            for label in labels:
                if config.dataset_name == "cifar10":
                    class_name = self.cifar10_classes.get(label, "object")
                    captions.append(f"a photo of a {class_name}")
                else:
                    captions.append(f"a photo of class {label}")
        else:
            captions = [""] * len(pixel_values)

        # 3. Tokenize
        inputs = self.tokenizer(
            captions, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        result = {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids
        }
        
        # 如果有 label，也返回它，用于训练时的 Text Embedding 缓存优化
        if self.label_col and self.label_col in examples:
            result["labels"] = examples[self.label_col]
            
        return result

    def load_data(self):
        print(f"⬇️ 正在加载数据集: {config.dataset_name} ...")
        dataset = load_dataset(config.dataset_name, split="train", cache_dir=config.dataset_cache_dir)
        
        # 识别数据集列名
        column_names = dataset.column_names
        self.image_col = "image" if "image" in column_names else "img"
        self.text_col = "text" if "text" in column_names else None
        self.label_col = "label" if "label" in column_names else None
        
        print(f"📋 检测到列名: {column_names}")
        print(f"   Image列: {self.image_col}, Text列: {self.text_col}, Label列: {self.label_col}")

        # 使用 with_transform (set_transform) 动态处理
        # 传入 bound method self._transform_function 是可以 picklable 的
        dataset.set_transform(self._transform_function)
        
        # 内存缓存优化 (In-Memory Cache)
        if hasattr(config, "in_memory_cache") and config.in_memory_cache:
            if config.dataset_name == "cifar10":
                print("⚡️ 正在将 CIFAR-10 全量加载到内存 (In-Memory Cache) ...")
                # 遍历一次 dataset，触发 transform，并堆叠为 Tensor
                # 注意: 这会消耗约 700MB 内存，但会极大加速训练
                # 由于 dataset 设置了 transform，迭代它就会得到处理后的 Tensor
                
                # 为了加速，我们可以用 DataLoader 多进程来做这个预加载
                temp_loader = DataLoader(
                    dataset, 
                    batch_size=1000, 
                    num_workers=config.dataloader_num_workers,
                    shuffle=False
                )
                
                cached_data = []
                from tqdm import tqdm
                for batch in tqdm(temp_loader, desc="Caching Dataset"):
                    cached_data.append(batch)
                
                # 构造一个新的 TensorDataset
                # 我们需要合并 list of batches
                # batch 是一个 dict: {'pixel_values': Tensor, 'input_ids': Tensor, 'labels': Tensor}
                
                final_pixel_values = torch.cat([b["pixel_values"] for b in cached_data], dim=0)
                final_input_ids = torch.cat([b["input_ids"] for b in cached_data], dim=0)
                final_labels = torch.cat([b["labels"] for b in cached_data], dim=0) if "labels" in cached_data[0] else None
                
                print(f"✅ 缓存完成! Images: {final_pixel_values.shape}, RAM占用: {final_pixel_values.element_size() * final_pixel_values.numel() / 1e6:.2f} MB")
                
                # 创建一个简单的 TensorDatasetWrapper (必须是可 picklable 的，所以定义在外面)
                return InMemoryDataset(final_pixel_values, final_input_ids, final_labels)

        return dataset

# 必须定义在模块级别，以便 multiprocessing 可以 pickle
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, pixel_values, input_ids, labels=None):
        self.pixel_values = pixel_values
        self.input_ids = input_ids
        self.labels = labels
        
    def __len__(self):
        return len(self.pixel_values)
    
    def __getitem__(self, idx):
        item = {
            "pixel_values": self.pixel_values[idx],
            "input_ids": self.input_ids[idx]
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

def get_dataloader():
    dataset_handler = TextToImageDataset()
    dataset = dataset_handler.load_data()
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers, 
        persistent_workers=config.dataloader_persistent_workers,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    print("🧪 测试数据加载模块 (CIFAR-10 适配版)...")
    loader = get_dataloader()
    
    try:
        batch = next(iter(loader))
        print(f"\n✅ 数据加载成功!")
        print(f"📦 Image Batch Shape: {batch['pixel_values'].shape}")
        print(f"📝 Text Token Shape: {batch['input_ids'].shape}")
        
        # 打印第一个样本的 caption (反解 token)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        first_caption = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        print(f"🔍 样本 0 文本: '{first_caption}'")
        
        # 保存一张样本图用于验证
        import torchvision
        img = batch['pixel_values'][0] * 0.5 + 0.5 # 反归一化
        torchvision.utils.save_image(img, "sample_cifar10_resized.png")
        print(f"🖼️ 已保存样本图: sample_cifar10_resized.png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 数据加载失败: {e}")
