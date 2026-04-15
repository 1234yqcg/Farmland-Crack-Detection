import os
from pathlib import Path
from collections import Counter

data_root = Path(r'd:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection\data')

splits = ['train', 'val', 'test']

class_counter = Counter()

for split in splits:
    label_dir = data_root / split / 'labels'
    if not label_dir.exists():
        print(f"{split} labels not found")
        continue
    
    print(f"\n=== {split} ===")
    file_count = 0
    for label_file in label_dir.glob('*.txt'):
        file_count += 1
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
    
    print(f"Total files: {file_count}")

print("\n=== Class Distribution (All splits) ===")
for class_id in sorted(class_counter.keys()):
    print(f"Class {class_id}: {class_counter[class_id]} samples")

print(f"\nTotal samples: {sum(class_counter.values())}")
