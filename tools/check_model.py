import torch
model_path = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\outputs\exp_anchor_loss\weights\best.pt'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print("Keys:", checkpoint.keys())
if isinstance(checkpoint, dict):
    for k in checkpoint.keys():
        v = checkpoint[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: {type(v)}, shape={v.shape}")
        else:
            print(f"  {k}: {type(v)}")
