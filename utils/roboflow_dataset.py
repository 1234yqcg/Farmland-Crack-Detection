# Roboflow数据集适配器

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from typing import Dict, List, Tuple, Optional
from PIL import Image
import json
from collections import Counter

class RoboflowFarmlandDataset(Dataset):
    """
    农田干裂检测专用Roboflow数据集适配器
    
    支持三个类别：
    - 0: mild (轻度干裂)
    - 1: moderate (中度干裂) 
    - 2: severe (重度干裂)
    """
    
    def __init__(self, 
                 data_yaml_path: str,
                 split: str = "train",
                 image_size: Tuple[int, int] = (512, 512),
                 transform: Optional[A.Compose] = None,
                 augment: bool = True,
                 cache_images: bool = False):
        """
        初始化数据集
        
        Args:
            data_yaml_path: data.yaml文件路径
            split: 数据分割 ('train', 'val', 'test')
            image_size: 目标图像尺寸 (height, width)
            transform: 自定义数据增强
            augment: 是否启用数据增强
            cache_images: 是否缓存图像到内存（小数据集可用）
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.cache_images = cache_images
        
        # 解析配置文件
        self.config = self._load_yaml_config()
        
        # 获取类别信息
        self.num_classes = self.config['nc']
        names = self.config['names']
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names.keys(), key=int)]
        self.class_names = names
        
        # 验证类别设置
        self._validate_classes()
        
        # 设置数据集路径
        self.dataset_root = self.data_yaml_path.parent
        split_path = self.config.get(split)
        if split_path:
            split_path = Path(split_path)
            if not split_path.is_absolute():
                split_path = (self.dataset_root / split_path).resolve()
            self.images_dir = split_path
            if self.images_dir.name != "images":
                self.images_dir = self.images_dir / "images"
        else:
            self.images_dir = self.dataset_root / split / "images"
        self.labels_dir = self.images_dir.parent / "labels"
        
        # 验证目录结构
        self._validate_directory_structure()
        
        # 获取文件列表
        self.image_files = self._get_image_files()
        self.label_files = self._get_label_files()
        
        # 图像缓存
        self.image_cache = {} if cache_images else None
        
        # 统计信息
        self.stats = self._calculate_statistics()
        
        # 数据增强配置
        self.transform = transform if transform else self._get_default_transforms()
        
        print(f"📊 {split}集初始化完成:")
        print(f"  📸 图像数量: {len(self.image_files)}")
        print(f"  🏷️  标注数量: {len(self.label_files)}")
        print(f"  📋 类别: {self.class_names}")
        print(f"  📦 总标注框: {self.stats['total_boxes']}")
        print(f"  🎯 平均每图标注: {self.stats['avg_boxes_per_image']:.2f}")
    
    def _load_yaml_config(self) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证必需字段
            required_fields = ['nc', 'names']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"缺少必需字段: {field}")
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def _validate_classes(self):
        """验证类别设置"""
        print(f"📋 检测到 {len(self.class_names)} 个类别:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        
        if len(self.class_names) != self.num_classes:
            print(f"⚠️  注意: names数量({len(self.class_names)}) 与 nc({self.num_classes}) 不一致")
    
    def _validate_directory_structure(self):
        """验证目录结构"""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")
    
    def _get_image_files(self) -> List[Path]:
        """获取图像文件列表"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise RuntimeError(f"在 {self.images_dir} 中没有找到图像文件")
        
        # Windows不区分大小写，需要去重
        seen = set()
        unique_files = []
        for f in image_files:
            stem = f.stem.lower()
            if stem not in seen:
                seen.add(stem)
                unique_files.append(f)
        
        return sorted(unique_files)
    
    def _get_label_files(self) -> List[Optional[Path]]:
        """获取标签文件列表"""
        label_files = []
        
        for image_file in self.image_files:
            label_file = self.labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                label_files.append(label_file)
            else:
                print(f"⚠️  警告: 缺少标签文件 {label_file}")
                label_files.append(None)
        
        return label_files
    
    def _calculate_statistics(self) -> Dict:
        """计算数据集统计信息"""
        total_boxes = 0
        class_counts = Counter()
        
        for label_file in self.label_files:
            if label_file is None:
                continue
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < self.num_classes:
                                class_counts[class_id] += 1
                                total_boxes += 1
            except Exception as e:
                print(f"⚠️  警告: 读取标签文件失败 {label_file}: {e}")
        
        return {
            'total_boxes': total_boxes,
            'class_distribution': dict(class_counts),
            'avg_boxes_per_image': total_boxes / max(len(self.image_files), 1),
            'num_images_with_labels': sum(1 for label_file in self.label_files if label_file is not None)
        }
    
    def _get_default_transforms(self) -> A.Compose:
        """获取默认的数据增强配置"""
        if self.split == "train" and self.augment:
            # 训练集数据增强
            return A.Compose([
                # 基础变换
                A.Resize(self.image_size[0], self.image_size[1]),
                
                # 几何变换
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5
                ),
                
                # 颜色变换（适合农田图像）
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.CLAHE(clip_limit=2.0, p=0.3),
                
                # 噪声和模糊
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                
                # 标准化
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_area=16,
                min_visibility=0.1
            ))
        else:
            # 验证/测试集变换
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels']
            ))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个数据项"""
        
        # 获取文件路径
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        # 加载图像
        if self.cache_images and str(image_path) in self.image_cache:
            image = self.image_cache[str(image_path)]
        else:
            image = self._load_image(image_path)
            if self.cache_images:
                self.image_cache[str(image_path)] = image
        
        original_height, original_width = image.shape[:2]
        
        # 加载标签
        labels = self._load_labels(label_path, original_width, original_height)
        
        # 应用数据增强
        if len(labels) > 0:
            bboxes = [label[:4] for label in labels]
            class_labels = [int(label[4]) for label in labels]
        else:
            bboxes = []
            class_labels = []
        
        # 应用变换
        if self.transform:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
                # 确保转换为tensor
                if not isinstance(image, torch.Tensor):
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                print(f"⚠️  数据增强失败 {image_path}: {e}")
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # 基本处理
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 准备标签张量
        if len(bboxes) > 0:
            labels_tensor = torch.zeros((len(bboxes), 6))  # [batch_idx, class, x, y, w, h]
            for i, (bbox, class_label) in enumerate(zip(bboxes, class_labels)):
                labels_tensor[i] = torch.tensor([0, class_label, *bbox])
        else:
            labels_tensor = torch.zeros((0, 6))
        
        return {
            'image': image,
            'labels': labels_tensor,
            'image_path': str(image_path),
            'original_size': (original_width, original_height),
            'num_objects': len(bboxes)
        }
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """加载图像"""
        try:
            # 优先使用PIL读取（兼容性更好）
            image = Image.open(image_path)
            
            # 转换为RGB（如果是RGBA等其他格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image = np.array(image)
            
            return image
            
        except Exception as e:
            # 如果PIL失败，尝试OpenCV
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"无法读取图像: {image_path}")
                
                # BGR转RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            except Exception as e2:
                raise RuntimeError(f"加载图像失败 {image_path}: {e} (PIL) / {e2} (OpenCV)")
    
    def _load_labels(self, label_path: Optional[Path], img_width: int, img_height: int) -> List[List[float]]:
        """加载标签文件"""
        labels = []
        
        if label_path is None or not label_path.exists():
            return labels
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        # 标签格式错误，静默忽略
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 验证类别ID
                        if class_id < 0 or class_id >= self.num_classes:
                            # 类别ID超出范围，静默忽略
                            continue
                        
                        # 验证坐标范围
                        if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                            # 坐标值超出范围，静默忽略
                            continue
                        
                        # YOLO格式转换为像素坐标
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height
                        
                        # 转换为边界框格式 [x1, y1, x2, y2, class_id]
                        x1 = x_center_px - width_px / 2
                        y1 = y_center_px - height_px / 2
                        x2 = x_center_px + width_px / 2
                        y2 = y_center_px + height_px / 2
                        
                        # 确保边界框在图像范围内
                        x1 = max(0, min(x1, img_width))
                        y1 = max(0, min(y1, img_height))
                        x2 = max(0, min(x2, img_width))
                        y2 = max(0, min(y2, img_height))
                        
                        # 过滤太小的边界框
                        if (x2 - x1) < 2 or (y2 - y1) < 2:
                            continue
                        
                        labels.append([x1, y1, x2, y2, class_id])
                        
                    except ValueError as e:
                        print(f"⚠️  警告: 解析标签失败 {label_path}:{line_num}: {e}")
                        continue
        
        except Exception as e:
            print(f"⚠️  警告: 读取标签文件失败 {label_path}: {e}")
        
        return labels
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        class_counts = self.stats['class_distribution']
        
        # 如果没有标注数据，返回均匀权重
        if not class_counts:
            return torch.ones(self.num_classes)
        
        # 计算权重（逆频率）
        total_samples = sum(class_counts.values())
        weights = torch.zeros(self.num_classes)
        
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / (count * self.num_classes)
        
        # 归一化权重
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """可视化单个样本"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        sample = self[idx]
        image = sample['image']
        labels = sample['labels']
        
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # 反标准化
        if image.max() <= 1:
            image = image * 255
        image = image.astype(np.uint8)
        
        # 创建可视化
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # 绘制标注框
        colors = ['green', 'orange', 'red']  # 轻度、中度、重度
        
        for label in labels:
            if len(label) >= 6:
                class_id = int(label[1])
                x1, y1, x2, y2 = label[2:6]
                
                if 0 <= class_id < len(colors):
                    color = colors[class_id]
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                else:
                    color = 'blue'
                    class_name = f"unknown_{class_id}"
                
                # 绘制矩形框
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, 
                               facecolor='none', linestyle='-')
                ax.add_patch(rect)
                
                # 添加标签
                ax.text(x1, y1-5, class_name, color=color, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_title(f"样本 {idx}: {Path(sample['image_path']).name}\n"
                    f"原始尺寸: {sample['original_size']}, "
                    f"目标数量: {sample['num_objects']}")
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 可视化保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()

def create_farmland_dataloaders(data_yaml_path: str,
                               batch_size: int = 4,
                               num_workers: int = 4,
                               image_size: Tuple[int, int] = (512, 512),
                               train_split: str = "train",
                               val_split: str = "val",
                               test_split: str = "test",
                               cache_images: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建农田干裂检测数据加载器
    
    Args:
        data_yaml_path: data.yaml文件路径
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        image_size: 图像尺寸
        train_split: 训练集分割名称
        val_split: 验证集分割名称
        test_split: 测试集分割名称
        cache_images: 是否缓存图像
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建训练数据集
    train_dataset = RoboflowFarmlandDataset(
        data_yaml_path=data_yaml_path,
        split=train_split,
        image_size=image_size,
        augment=True,
        cache_images=cache_images
    )
    
    # 创建验证数据集
    val_dataset = RoboflowFarmlandDataset(
        data_yaml_path=data_yaml_path,
        split=val_split,
        image_size=image_size,
        augment=False,
        cache_images=cache_images
    )
    
    # 创建测试数据集
    test_dataset = RoboflowFarmlandDataset(
        data_yaml_path=data_yaml_path,
        split=test_split,
        image_size=image_size,
        augment=False,
        cache_images=cache_images
    )
    
    # 创建数据加载器
    def collate_fn(batch):
        """自定义批次合并函数"""
        images = torch.stack([item['image'] for item in batch])
        
        # 合并所有标签
        all_labels = []
        for i, item in enumerate(batch):
            labels = item['labels'].clone()
            if len(labels) > 0:
                labels[:, 0] = i  # 设置批次索引
                all_labels.append(labels)
        
        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)
        else:
            all_labels = torch.zeros((0, 6))
        
        return {
            'images': images,
            'labels': all_labels,
            'image_paths': [item['image_path'] for item in batch],
            'original_sizes': [item['original_size'] for item in batch],
            'num_objects': [item['num_objects'] for item in batch]
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

# 使用示例和测试函数
def test_dataset():
    """测试数据集功能"""
    print("🧪 测试Roboflow农田数据集...")
    
    # 测试数据集路径
    data_yaml_path = "data/roboflow/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ 测试失败: 找不到数据集配置文件 {data_yaml_path}")
        return False
    
    try:
        # 创建数据集
        dataset = RoboflowFarmlandDataset(
            data_yaml_path=data_yaml_path,
            split="train",
            image_size=(512, 512),
            augment=True,
            cache_images=False
        )
        
        print(f"✅ 数据集创建成功")
        print(f"📊 数据集统计:")
        print(f"  📸 图像数量: {len(dataset)}")
        print(f"  📦 标注框总数: {dataset.stats['total_boxes']}")
        print(f"  📋 类别分布: {dataset.stats['class_distribution']}")
        
        # 测试数据加载
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n✅ 样本加载成功:")
            print(f"  🖼️  图像形状: {sample['image'].shape}")
            print(f"  🏷️  标签数量: {len(sample['labels'])}")
            print(f"  📁 图像路径: {sample['image_path']}")
        
        # 测试类别权重
        class_weights = dataset.get_class_weights()
        print(f"\n✅ 类别权重: {class_weights}")
        
        # 可视化样本（可选）
        if len(dataset) > 0:
            dataset.visualize_sample(0, save_path="test_sample.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    # 运行测试
    success = test_dataset()
    
    if success:
        print("\n🎉 数据集测试通过！")
    else:
        print("\n❌ 数据集测试失败！")
        exit(1)
