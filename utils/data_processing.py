import cv2
import numpy as np
from typing import Tuple, Optional

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        target_h, target_w = self.target_size
        
        scale = min(target_w / width, target_h / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0
    
    def convert_color_space(self, image: np.ndarray, 
                           src_space: str = 'BGR', 
                           dst_space: str = 'RGB') -> np.ndarray:
        if src_space == 'BGR' and dst_space == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif src_space == 'RGB' and dst_space == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def resize_with_padding(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        transform_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'original_size': (w, h),
            'new_size': (new_w, new_h)
        }
        
        return canvas, transform_info
    
    def color_correction(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def denoise(self, image: np.ndarray, method: str = 'fast') -> np.ndarray:
        if method == 'fast':
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (3, 3), 0)
        return image
    
    def preprocess_pipeline(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        corrected = self.color_correction(image)
        denoised = self.denoise(corrected, method='fast')
        resized, transform_info = self.resize_with_padding(denoised)
        
        return resized, transform_info
