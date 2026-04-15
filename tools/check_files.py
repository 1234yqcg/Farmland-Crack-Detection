import os
from pathlib import Path

def check_dataset_files():
    source_dir = Path(r"d:\桌面\学习\000毕业设计！\YOLOv10\Drought.v3-originalus-70-20-10-bet-vienas-paveikslelis-turi-.yolov9")
    
    print("检查训练集文件...")
    train_images_dir = source_dir / "train" / "images"
    
    files = list(train_images_dir.glob("*.jpg"))
    print(f"找到 {len(files)} 个 .jpg 文件\n")
    
    print("前10个文件信息:")
    for i, file_path in enumerate(files[:10]):
        size = file_path.stat().st_size if file_path.exists() else 0
        print(f"{i+1}. {file_path.name}")
        print(f"   大小: {size} bytes ({size/1024:.2f} KB)")
        print(f"   存在: {file_path.exists()}")
        print()

if __name__ == "__main__":
    check_dataset_files()