import os
import shutil
from pathlib import Path
import cv2

def migrate_and_clean_dataset():
    source_dir = Path(r"d:\桌面\学习\000毕业设计！\YOLOv10\Drought.v3-originalus-70-20-10-bet-vienas-paveikslelis-turi-.yolov9")
    target_dir = Path(r"d:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\data")
    
    # 清空目标目录
    print("🗑️  清空目标目录...")
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            target_path = target_dir / split / subdir
            if target_path.exists():
                shutil.rmtree(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
    
    # 迁移训练集
    print("\n📦 迁移训练集...")
    train_images_dir = source_dir / "train" / "images"
    train_labels_dir = source_dir / "train" / "labels"
    
    valid_train_count = 0
    corrupted_train_count = 0
    
    for img_path in train_images_dir.glob("*.jpg"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                # 复制图像
                shutil.copy2(img_path, target_dir / "train" / "images" / img_path.name)
                # 复制标签
                label_path = train_labels_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    shutil.copy2(label_path, target_dir / "train" / "labels" / label_path.name)
                valid_train_count += 1
            else:
                corrupted_train_count += 1
                print(f"❌ 损坏: {img_path.name}")
        except Exception as e:
            corrupted_train_count += 1
            print(f"❌ 错误: {img_path.name} - {e}")
    
    # 迁移验证集
    print("\n📦 迁移验证集...")
    val_images_dir = source_dir / "valid" / "images"
    val_labels_dir = source_dir / "valid" / "labels"
    
    valid_val_count = 0
    corrupted_val_count = 0
    
    for img_path in val_images_dir.glob("*.jpg"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                # 复制图像
                shutil.copy2(img_path, target_dir / "val" / "images" / img_path.name)
                # 复制标签
                label_path = val_labels_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    shutil.copy2(label_path, target_dir / "val" / "labels" / label_path.name)
                valid_val_count += 1
            else:
                corrupted_val_count += 1
                print(f"❌ 损坏: {img_path.name}")
        except Exception as e:
            corrupted_val_count += 1
            print(f"❌ 错误: {img_path.name} - {e}")
    
    # 迁移测试集
    print("\n📦 迁移测试集...")
    test_images_dir = source_dir / "test" / "images"
    test_labels_dir = source_dir / "test" / "labels"
    
    valid_test_count = 0
    corrupted_test_count = 0
    
    for img_path in test_images_dir.glob("*.jpg"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                # 复制图像
                shutil.copy2(img_path, target_dir / "test" / "images" / img_path.name)
                # 复制标签
                label_path = test_labels_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    shutil.copy2(label_path, target_dir / "test" / "labels" / label_path.name)
                valid_test_count += 1
            else:
                corrupted_test_count += 1
                print(f"❌ 损坏: {img_path.name}")
        except Exception as e:
            corrupted_test_count += 1
            print(f"❌ 错误: {img_path.name} - {e}")
    
    # 统计结果
    print(f"\n📊 迁移完成！")
    print(f"训练集: {valid_train_count} 张有效图像, {corrupted_train_count} 张损坏")
    print(f"验证集: {valid_val_count} 张有效图像, {corrupted_val_count} 张损坏")
    print(f"测试集: {valid_test_count} 张有效图像, {corrupted_test_count} 张损坏")
    print(f"总计: {valid_train_count + valid_val_count + valid_test_count} 张有效图像")

if __name__ == "__main__":
    migrate_and_clean_dataset()