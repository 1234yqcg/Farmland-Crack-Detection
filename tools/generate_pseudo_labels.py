# 功能：用当前最佳模型对未标注数据生成伪标签（Self-Training / Pseudo-Labeling）
# 1. 用 v8 best.pt 对 China_Drone 全部 2401 张图片推理
# 2. 筛选置信度 >0.5 的检测结果，保存为 YOLO 格式标签
# 3. 输出伪标签图片和标签到 data/pseudo_china_drone/
# 4. 自动生成 dataset_pseudo.yaml 供训练使用
# 该方法可在不增加手工标注的情况下大幅扩增训练集规模
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.yolov10_crack import YOLOv10Crack
from evaluate import decode_predictions

# ---------- 配置 ----------
V8_WEIGHTS = 'outputs/farmland_finetune_v8/weights/best.pt'
CHINA_DIR = Path('D:/桌面/学习/000毕业设计！/YOLOv10/China_Drone/train/images')
OUTPUT_DIR = Path('data/pseudo_china_drone')  # 伪标签输出目录
CONF_THRESHOLD = 0.5  # 伪标签置信度阈值
NMS_IOU = 0.5
IMG_SIZE = 800

# ---------- 准备 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = YOLOv10Crack(num_classes=3).to(device)
ckpt = torch.load(V8_WEIGHTS, map_location=device)
state_dict = ckpt.get('ema_model', ckpt.get('model', ckpt))
if hasattr(state_dict, 'state_dict'):
    state_dict = state_dict.state_dict()
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"Model loaded: {V8_WEIGHTS}")

out_img_dir = OUTPUT_DIR / "images"
out_lbl_dir = OUTPUT_DIR / "labels"
out_img_dir.mkdir(parents=True, exist_ok=True)
out_lbl_dir.mkdir(parents=True, exist_ok=True)

imgs = sorted(CHINA_DIR.glob('*.jpg'))
print(f"Total China_Drone images: {len(imgs)}")

# ---------- 统计 ----------
stats = {
    'processed': 0,
    'images_with_preds': 0,
    'images_with_high': 0,  # >0.7
    'total_boxes': 0,
    'high_conf_boxes': 0,
    'class_counts': {0: 0, 1: 0, 2: 0},
    'class_counts_high': {0: 0, 1: 0, 2: 0},  # >0.7
}

# ---------- 推理 ----------
for img_path in tqdm(imgs, desc="Pseudo-labeling all"):
    try:
        img = Image.open(str(img_path)).convert('RGB')
        orig_w, orig_h = img.size

        # 缩放到统一尺寸
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        dets = decode_predictions(outputs[0], conf_threshold=CONF_THRESHOLD, iou_threshold=NMS_IOU)
    except Exception as e:
        continue

    stats['processed'] += 1

    if not dets:
        continue

    stats['images_with_preds'] += 1
    stats['total_boxes'] += len(dets)

    # 筛选高置信度
    high_dets = [d for d in dets if d['score'] > 0.7]
    if high_dets:
        stats['images_with_high'] += 1
        stats['high_conf_boxes'] += len(high_dets)

    # 输出伪标签（YOLO格式）
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE

    label_lines = []
    for d in dets:
        cls_id = d['class_id']
        score = d['score']
        stats['class_counts'][cls_id] = stats['class_counts'].get(cls_id, 0) + 1
        if score > 0.7:
            stats['class_counts_high'][cls_id] = stats['class_counts_high'].get(cls_id, 0) + 1

        # bbox: [x1, y1, x2, y2] pixel coords
        x1, y1, x2, y2 = d['bbox']
        # 缩放到原图尺寸
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        # 转YOLO: cx, cy, w, h (normalized)
        cx = ((x1 + x2) / 2) / orig_w
        cy = ((y1 + y2) / 2) / orig_h
        bw = (x2 - x1) / orig_w
        bh = (y2 - y1) / orig_h
        # 裁剪到[0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if label_lines:
        # 保存标签
        lbl_out = out_lbl_dir / f"{img_path.stem}.txt"
        with open(lbl_out, 'w') as f:
            f.write('\n'.join(label_lines))
        # 复制图片（使用创建链接节省空间）
        img_out = out_img_dir / f"{img_path.stem}.jpg"
        if not img_out.exists():
            img.save(str(img_out), quality=95)

# ---------- 报告 ----------
CLASS_NAMES = ['细微裂纹', '网状裂隙', '深大裂缝']
print(f"\n{'='*60}")
print(f"伪标签全量推理报告")
print(f"{'='*60}")
print(f"处理图片: {stats['processed']}/{len(imgs)}")
print(f"含预测框(>{CONF_THRESHOLD})的图片: {stats['images_with_preds']} ({stats['images_with_preds']/max(stats['processed'],1)*100:.1f}%)")
print(f"含高置信度(>0.7)的图片: {stats['images_with_high']} ({stats['images_with_high']/max(stats['processed'],1)*100:.1f}%)")
print(f"")
print(f"总预测框数: {stats['total_boxes']}")
print(f"高置信度框数(>0.7): {stats['high_conf_boxes']}")
print(f"")
print(f"各类别框数:")
for cid in range(3):
    n = stats['class_counts'].get(cid, 0)
    nh = stats['class_counts_high'].get(cid, 0)
    print(f"  {cid} ({CLASS_NAMES[cid]}): 总计 {n} 框, 高置信度 {nh} 框")
print(f"")
print(f"保存路径: {OUTPUT_DIR.resolve()}")
print(f"{'='*60}")

# ---------- 生成 dataset_pseudo.yaml ----------
yaml_content = f"""train: {OUTPUT_DIR}
val: ./data/val
test: ./data/test
nc: 3
names: ['细微裂纹', '网状裂隙', '深大裂缝']
"""
yaml_path = OUTPUT_DIR.parent / 'dataset_pseudo.yaml'
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)
print(f"Dataset config saved: {yaml_path}")
