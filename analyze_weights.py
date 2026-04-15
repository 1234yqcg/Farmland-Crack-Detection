import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "Farmland_Crack_Detection"))
from models.yolov10_crack import YOLOv10Crack

print("=" * 60)
print("分析官方 yolov10s.pt 的层名称")
print("=" * 60)

from ultralytics import YOLO
yolo_model = YOLO("./weights/yolov10s.pt")
pretrained_state = yolo_model.model.state_dict()

print("\n官方 yolov10s 层名称 (前30个):")
for i, (name, param) in enumerate(pretrained_state.items()):
    if i < 30:
        print(f"  {name}: {param.shape}")
    else:
        break

print("\n" + "=" * 60)
print("分析自定义 YOLOv10Crack 的层名称")
print("=" * 60)

model = YOLOv10Crack(num_classes=3, reg_max=16)
model_state = model.state_dict()

print("\n自定义模型层名称 (前30个):")
for i, (name, param) in enumerate(model_state.items()):
    if i < 30:
        print(f"  {name}: {param.shape}")
    else:
        break

print("\n" + "=" * 60)
print("匹配分析")
print("=" * 60)

matched = []
for pname, pparam in pretrained_state.items():
    for mname, mparam in model_state.items():
        if pparam.shape == mparam.shape:
            if 'cv2' not in pname and 'head' not in mname.lower():
                matched.append((pname, mname, pparam.shape))

print(f"\n找到 {len(matched)} 个可能匹配的层:")
for pname, mname, shape in matched[:20]:
    print(f"  官方: {pname}")
    print(f"  自定义: {mname}")
    print(f"  形状: {shape}")
    print()
