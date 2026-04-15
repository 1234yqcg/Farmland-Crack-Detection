import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0, r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection')

from models.yolov10_crack import YOLOv10Crack

def nms(boxes, scores, iou_threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def load_yolo_labels(label_path, img_width, img_height):
    boxes = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    boxes.append([x1, y1, x2, y2])
    return boxes

model_path = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\outputs\exp_yolov10s\weights\best.pt'
test_dir = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\data\test\images'
label_dir = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\data\test\labels'
output_dir = r'd:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection\outputs\inference_results'

conf_threshold = 0.15
iou_threshold = 0.3

Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model_state = checkpoint['model']

model = YOLOv10Crack(
    num_classes=1, 
    depth_multiple=0.33,
    width_multiple=0.50,
    use_attention=True,
    reg_max=16
)

model_dict = model.state_dict()
filtered_state = {}
for k, v in model_state.items():
    if k in model_dict and v.shape == model_dict[k].shape:
        filtered_state[k] = v

model.load_state_dict(filtered_state, strict=False)
model.eval()

test_images = sorted(Path(test_dir).glob('*.jpg'))
print(f"\nFound {len(test_images)} test images")

total_gt = 0
total_pred = 0
results = []

for img_path in test_images:
    img_name = img_path.stem
    label_path = Path(label_dir) / f"{img_name}.txt"
    output_path = Path(output_dir) / f"{img_name}_result.jpg"
    
    pil_img = Image.open(img_path).convert('RGB')
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    original_h, original_w = image.shape[:2]
    
    input_size = (512, 512)
    resized = cv2.resize(image, input_size)
    normalized = resized.astype(np.float32) / 255.0
    normalized = np.transpose(normalized, (2, 0, 1))
    normalized = np.expand_dims(normalized, 0)
    
    with torch.no_grad():
        outputs = model(torch.from_numpy(normalized))
    
    outputs = outputs.squeeze(0).numpy()
    
    boxes = outputs[:, :4]
    scores = outputs[:, 4]
    
    high_conf_mask = scores > conf_threshold
    high_conf_boxes = boxes[high_conf_mask]
    high_conf_scores = scores[high_conf_mask]
    
    keep = nms(high_conf_boxes, high_conf_scores, iou_threshold=iou_threshold)
    
    scale_factor_x = original_w / input_size[1]
    scale_factor_y = original_h / input_size[0]
    
    gt_boxes = load_yolo_labels(label_path, original_w, original_h)
    total_gt += len(gt_boxes)
    total_pred += len(keep)
    
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, 'GT', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    for idx in keep:
        x1, y1, x2, y2 = high_conf_boxes[idx]
        conf = high_conf_scores[idx]
        x1 = int(x1 * scale_factor_x)
        y1 = int(y1 * scale_factor_y)
        x2 = int(x2 * scale_factor_x)
        y2 = int(y2 * scale_factor_y)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Pred:{conf:.2f}'
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_result = Image.fromarray(image_rgb)
    pil_result.save(str(output_path), quality=95)
    
    results.append({
        'image': img_name,
        'gt': len(gt_boxes),
        'pred': len(keep)
    })
    print(f"{img_name}: GT={len(gt_boxes)}, Pred={len(keep)}")

print(f"\n{'='*50}")
print(f"Total Summary:")
print(f"{'='*50}")
print(f"Total GT boxes: {total_gt}")
print(f"Total Pred boxes: {total_pred}")
print(f"Results saved to: {output_dir}")
