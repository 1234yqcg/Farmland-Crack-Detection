import os
import shutil
import random
from pathlib import Path

def split_dataset():
    data_dir = Path(r"d:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\data")
    train_images_dir = data_dir / "train" / "images"
    train_labels_dir = data_dir / "train" / "labels"
    val_images_dir = data_dir / "val" / "images"
    val_labels_dir = data_dir / "val" / "labels"
    
    # 确保验证集目录存在
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有训练图像
    train_images = list(train_images_dir.glob("*.jpg"))
    train_images.extend(list(train_images_dir.glob("*.jpeg")))
    train_images.extend(list(train_images_dir.glob("*.png")))
    
    print(f"训练集总图像数: {len(train_images)}")
    
    # 随机选择20%作为验证集
    val_count = max(1, int(len(train_images) * 0.2))
    val_images = random.sample(train_images, val_count)
    
    print(f"移动到验证集的图像数: {val_count}")
    
    # 移动图像和对应的标签
    for img_path in val_images:
        # 移动图像
        shutil.move(str(img_path), str(val_images_dir / img_path.name))
        
        # 移动对应的标签
        label_path = train_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels_dir / label_path.name))
            print(f"移动: {img_path.name} + {label_path.name}")
        else:
            print(f"警告: 找不到标签 {label_path.name}")
    
    # 统计最终结果
    final_train_count = len(list(train_images_dir.glob("*.jpg"))) + len(list(train_images_dir.glob("*.jpeg"))) + len(list(train_images_dir.glob("*.png")))
    final_val_count = len(list(val_images_dir.glob("*.jpg"))) + len(list(val_images_dir.glob("*.jpeg"))) + len(list(val_images_dir.glob("*.png")))
    
    print(f"\n最终统计:")
    print(f"训练集: {final_train_count} 张图像")
    print(f"验证集: {final_val_count} 张图像")
    print(f"测试集: {len(list((data_dir / 'test' / 'images').glob('*.jpg')))} 张图像")

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子保证可重复性
    split_dataset()