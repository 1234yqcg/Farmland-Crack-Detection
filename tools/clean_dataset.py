import os
import shutil
from pathlib import Path

data_root = Path(r'd:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection\data')
target_class = 2

for split in ['train', 'val', 'test']:
    image_dir = data_root / split / 'images'
    label_dir = data_root / split / 'labels'
    
    if not image_dir.exists():
        continue
    
    print(f"\n=== Processing {split} ===")
    
    kept_count = 0
    removed_count = 0
    
    for label_file in list(label_dir.glob('*.txt')):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        has_target_class = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id == target_class:
                    parts[0] = '0'
                    new_lines.append(' '.join(parts) + '\n')
                    has_target_class = True
        
        if has_target_class and new_lines:
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
            kept_count += 1
        else:
            image_name = label_file.stem
            label_file.unlink()
            
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                image_file = image_dir / (image_name + ext)
                if image_file.exists():
                    image_file.unlink()
                    break
            removed_count += 1
    
    print(f"Kept: {kept_count}, Removed: {removed_count}")

print("\n=== Dataset cleaned! ===")
