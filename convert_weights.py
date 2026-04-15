import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "Farmland_Crack_Detection"))
from models.yolov10_crack import YOLOv10Crack

from ultralytics import YOLO
yolo_model = YOLO("./weights/yolov10s.pt")
pretrained_state = yolo_model.model.state_dict()

model = YOLOv10Crack(num_classes=3, reg_max=16)
model_state = model.state_dict()

mapping = {
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage2.2',
    'model.6': 'backbone.stage3.0',
    'model.7': 'backbone.stage3.1',
    'model.8': 'backbone.stage3.2',
    'model.9': 'backbone.stage3.3',
    'model.10': 'backbone.stage3.4',
    'model.11': 'backbone.stage3.5',
    'model.12': 'backbone.stage4.0',
    'model.13': 'backbone.stage4.1',
    'model.14': 'backbone.stage4.2',
    'model.15': 'backbone.stage4.3',
    'model.16': 'backbone.stage4.4',
    'model.17': 'backbone.stage4.5',
    'model.18': 'backbone.stage4.6',
    'model.19': 'backbone.stage4.7',
    'model.20': 'neck.fpn_p5',
    'model.21': 'neck.fpn_p4',
    'model.22': 'neck.fpn_p3',
    'model.23': 'neck.pan_n3',
    'model.24': 'neck.pan_n4',
    'model.25': 'neck.pan_n5',
}

loaded = 0
new_state = model_state.copy()

for pname, pparam in pretrained_state.items():
    for old_prefix, new_prefix in mapping.items():
        if pname.startswith(old_prefix + '.'):
            new_name = pname.replace(old_prefix + '.', new_prefix + '.', 1)
            if new_name in model_state and model_state[new_name].shape == pparam.shape:
                new_state[new_name] = pparam
                loaded += 1
            break

print(f"成功加载 {loaded} 层")

model.load_state_dict(new_state)

save_path = Path("./weights/yolov10s_backbone.pt")
torch.save({'model': new_state}, save_path)
print(f"保存到: {save_path}")
