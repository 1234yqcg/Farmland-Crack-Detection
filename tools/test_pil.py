import os
from pathlib import Path
from PIL import Image

def test_image_reading():
    source_dir = Path(r"d:\桌面\学习\000毕业设计！\YOLOv10\Drought.v3-originalus-70-20-10-bet-vienas-paveikslelis-turi-.yolov9")
    train_images_dir = source_dir / "train" / "images"
    
    print("测试图像读取...\n")
    
    files = list(train_images_dir.glob("*.jpg"))[:10]
    
    for file_path in files:
        print(f"文件: {file_path.name}")
        
        # 尝试用PIL读取
        try:
            with Image.open(file_path) as img:
                print(f"  ✅ PIL读取成功")
                print(f"     格式: {img.format}")
                print(f"     尺寸: {img.size}")
                print(f"     模式: {img.mode}")
        except Exception as e:
            print(f"  ❌ PIL读取失败: {e}")
        
        print()

if __name__ == "__main__":
    test_image_reading()