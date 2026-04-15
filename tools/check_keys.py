import torch

model_path = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\outputs\exp_anchor_loss\weights\best.pt'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model_state = checkpoint['model']

print("Sample keys from checkpoint:")
keys = list(model_state.keys())
for k in keys[:30]:
    v = model_state[k]
    print(f"  {k}: {v.shape}")
print(f"\nTotal keys: {len(keys)}")
