import torch
import sys
sys.path.insert(0, r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection')

from models.yolov10_crack import YOLOv10Crack

model_path = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\outputs\exp_anchor_loss\weights\best.pt'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model_state = checkpoint['model']

model = YOLOv10Crack(
    num_classes=1, 
    depth_multiple=0.33,
    width_multiple=0.25,
    use_attention=True,
    reg_max=16
)

model_dict = model.state_dict()
filtered_state = {}
skipped = []
loaded = []
for k, v in model_state.items():
    if k in model_dict and v.shape == model_dict[k].shape:
        filtered_state[k] = v
        loaded.append(k)
    else:
        skipped.append((k, v.shape, model_dict.get(k, 'N/A')))

print(f"Loaded: {len(loaded)} layers")
print(f"Skipped: {len(skipped)} layers")

for k, v_shape, m_shape in skipped[:30]:
    print(f"  Skip: {k}, checkpoint: {v_shape}")
