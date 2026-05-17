import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


class RoboflowFarmlandDataset(Dataset):
    """
    农田裂缝检测数据集加载器
    
    支持Roboflow格式的YOLO数据集
    """
    
    def __init__(self,
                 data_yaml_path: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (640, 640),
                 augment: bool = True):
        """
        初始化数据集
        
        Args:
            data_yaml_path: data.yaml文件路径
            split: 数据集划分 ('train', 'val', 'test')
            image_size: 目标图像尺寸 (H, W)
            augment: 是否进行数据增强
        """
        self.data_yaml_path = data_yaml_path
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # 加载数据集配置
        self._load_config()
        
        # 获取图像和标签路径
        self.images, self.labels = _load_data_paths(
            self.data_root, 
            self.split,
            self.config.get('path', '')
        )
        
        # 类别信息
        self.num_classes = int(self.config['nc'])
        self.class_names = self._parse_class_names()
    
    def _load_config(self):
        """加载data.yaml配置"""
        import yaml
        
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 确定数据根目录
        if 'path' in self.config:
            base_path = self.config['path']
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(self.data_yaml_path), base_path)
            self.data_root = base_path
        else:
            self.data_root = os.path.dirname(self.data_yaml_path)
    
    def _parse_class_names(self) -> List[str]:
        """解析类别名称"""
        names = self.config.get('names', [])
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys(), key=int)]
        return names if isinstance(names, list) else [f'class_{i}' for i in range(self.num_classes)]
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            包含image和labels的字典
        """
        image_path = self.images[idx]
        label_path = self.labels[idx]
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        labels = self._read_labels(label_path)
        
        # 预处理图像
        image, labels = self._preprocess(image, labels)
        
        return {
            'image': torch.from_numpy(image).float(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
    
    def _read_labels(self, label_path: str) -> np.ndarray:
        """读取YOLO格式标签"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = float(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        labels.append([class_id, cx, cy, w, h])
        
        if not labels:
            labels = [[-1, 0, 0, 0, 0]]  # 无目标标记
        
        return np.array(labels, dtype=np.float32)
    
    def _preprocess(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        图像预处理和数据增强
        """
        h, w = image.shape[:2]
        target_h, target_w = self.image_size
        
        # 缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # 归一化到[0, 1]
        normalized = padded.astype(np.float32) / 255.0
        
        # HWC -> CHW
        normalized = normalized.transpose(2, 0, 1)
        
        # 调整标签坐标（考虑padding）
        if len(labels) > 0 and labels[0][0] != -1:
            adjusted_labels = labels.copy()
            adjusted_labels[:, 1] = (labels[:, 1] * w * scale + pad_x) / target_w
            adjusted_labels[:, 2] = (labels[:, 2] * h * scale + pad_y) / target_h
            adjusted_labels[:, 3] = labels[:, 3] * w * scale / target_w
            adjusted_labels[:, 4] = labels[:, 4] * h * scale / target_h
            labels = adjusted_labels
        
        return normalized, labels


def _load_data_paths(data_root: str, split: str, path_override: str = '') -> Tuple[List[str], List[str]]:
    """
    加载数据路径
    
    Args:
        data_root: 数据根目录
        split: 数据划分
        path_override: 路径覆盖（来自yaml配置）
    
    Returns:
        (images列表, labels列表)
    """
    # 使用path_override或data_root作为基础路径
    base_dir = path_override if path_override else data_root
    
    img_dir = os.path.join(base_dir, split, 'images')
    lbl_dir = os.path.join(base_dir, split, 'labels')
    
    images = []
    labels = []
    
    # 支持多种图像格式
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if os.path.exists(img_dir):
        for ext in extensions:
            for img_file in Path(img_dir).glob(f'*{ext}'):
                images.append(str(img_file))
                stem = img_file.stem
                labels.append(os.path.join(lbl_dir, stem + '.txt'))
    
    return images, labels
