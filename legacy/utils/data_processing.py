import cv2
import numpy as np
from typing import Dict, Tuple, Any


class ImagePreprocessor:
    """
    图像预处理器
    用于农田裂缝检测的图像预处理流程
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
    
    def resize_with_padding(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        等比例缩放图像并填充到目标尺寸
        
        Args:
            image: 输入图像 (H, W, C)
            
        Returns:
            处理后的图像和缩放信息字典
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # 计算缩放比例（保持长宽比）
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 计算padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        pad_x_right = target_w - new_w - pad_x
        pad_y_bottom = target_h - new_h - pad_y
        
        # 填充图像（使用常数填充，值为0）
        padded = cv2.copyMakeBorder(
            resized,
            pad_y, pad_y_bottom,
            pad_x, pad_x_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'original_size': (h, w),
            'resized_size': (new_h, new_w)
        }
        
        return padded, info
    
    def color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        颜色校正（CLAHE对比度增强）
        
        Args:
            image: 输入图像
            
        Returns:
            校正后的图像
        """
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 应用CLAHE到L通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # 合并通道并转换回BGR
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        corrected = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return corrected.astype(np.uint8)
    
    def preprocess_pipeline(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        完整的预处理流水线
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像和预处理信息
        """
        # 颜色校正
        corrected = self.color_correction(image)
        
        # 缩放和填充
        processed, info = self.resize_with_padding(corrected)
        
        return processed, info
