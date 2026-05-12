# 功能：Copy-Paste 实例级数据增强
# 从训练集中提取所有裂缝目标实例（抠图），然后随机粘贴到其他训练图像上，
# 实现目标数量的翻倍与背景多样性的增加，提升模型的泛化能力。
# 使用泊松融合（seamlessClone）使粘贴的实例边缘更自然。
# 输出：data/train_copypaste/（含原图+增强图，配合 dataset_copypaste.yaml 使用）
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import random

def get_instances(image_dir, label_dir):
    """提取所有的目标实例（裁剪的图像块和类别）"""
    instances = []
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for img_path in tqdm(image_files, desc="Extracting instances"):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        try:
            # OpenCV 处理中文路径
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            h, w = img.shape[:2]
        except:
            continue
            
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                        
                    instance_crop = img[y1:y2, x1:x2].copy()
                    instances.append({
                        'image': instance_crop,
                        'class_id': cls_id
                    })
    return instances

def copy_paste_augment(image_dir, label_dir, out_img_dir, out_lbl_dir, instances, num_augments_per_image=3):
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for img_path in tqdm(image_files, desc="Applying Copy-Paste"):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        
        try:
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            h, w = img.shape[:2]
        except:
            continue
            
        # 复制原图
        aug_img = img.copy()
        
        # 读取原标签
        labels = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                        
        # 随机选择实例进行粘贴
        num_pastes = random.randint(1, num_augments_per_image)
        for _ in range(num_pastes):
            if not instances:
                break
            inst = random.choice(instances)
            inst_img = inst['image']
            inst_cls = inst['class_id']
            
            ih, iw = inst_img.shape[:2]
            
            # 随机选择粘贴位置 (确保不超出边界)
            if w - iw <= 0 or h - ih <= 0:
                continue
                
            px = random.randint(0, w - iw)
            py = random.randint(0, h - ih)
            
            # 泊松融合 (Poisson Blending) 或者直接覆盖
            # 这里为了速度和稳定性，使用简单的羽化边缘覆盖
            mask = 255 * np.ones(inst_img.shape, inst_img.dtype)
            try:
                center = (px + iw // 2, py + ih // 2)
                aug_img = cv2.seamlessClone(inst_img, aug_img, mask, center, cv2.NORMAL_CLONE)
            except:
                # 如果 seamlessClone 失败，直接覆盖
                aug_img[py:py+ih, px:px+iw] = inst_img
                
            # 添加新标签
            new_cx = (px + iw / 2) / w
            new_cy = (py + ih / 2) / h
            new_bw = iw / w
            new_bh = ih / h
            labels.append([inst_cls, new_cx, new_cy, new_bw, new_bh])
            
        # 保存新图像和标签
        out_name = f"{img_path.stem}_cp.jpg"
        cv2.imencode('.jpg', aug_img)[1].tofile(str(out_img_dir / out_name))
        
        with open(out_lbl_dir / f"{img_path.stem}_cp.txt", "w") as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/train_copypaste")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent / args.data_dir
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    
    out_dir = Path(__file__).resolve().parent.parent / args.output_dir
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. 提取所有裂缝实例...")
    instances = get_instances(train_img_dir, train_lbl_dir)
    print(f"共提取 {len(instances)} 个实例。")
    
    print("2. 进行 Copy-Paste 数据增强...")
    copy_paste_augment(train_img_dir, train_lbl_dir, out_img_dir, out_lbl_dir, instances)
    
    print(f"增强完成，数据保存在 {out_dir}")

if __name__ == "__main__":
    main()
