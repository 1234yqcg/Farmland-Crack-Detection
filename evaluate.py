import os
import sys
import warnings
warnings.filterwarnings('ignore')

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from utils.metrics import calculate_map, calculate_precision_recall
from utils.roboflow_dataset import RoboflowFarmlandDataset


def box_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    box2_area = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1:
            break
        remaining = indices[1:]
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in remaining])
        indices = remaining[ious <= iou_threshold]
    return keep


def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        images.append(item['image'])
        targets.append(item['labels'])
    images = torch.stack(images, dim=0)
    return {'images': images, 'targets': targets}


def load_data_config(dataset_yaml):
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_default_data_yaml():
    return os.path.join(os.path.dirname(__file__), 'data', 'dataset.yaml')


def split_exists(data_yaml, data_config, split):
    split_path = data_config.get(split)
    if not split_path:
        return False
    if not os.path.isabs(split_path):
        split_path = os.path.join(os.path.dirname(data_yaml), split_path)
    if not os.path.exists(split_path):
        return False
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return any(name.lower().endswith(image_extensions) for name in os.listdir(split_path))


def build_model(weights_path, device, num_classes):
    model = YOLOv10Crack(num_classes=num_classes, reg_max=16).to(device)
    checkpoint_meta = {}
    if os.path.exists(weights_path):
        checkpoint = None
        last_error = None
        for weights_only in (True, False):
            try:
                checkpoint = torch.load(weights_path, map_location=device, weights_only=weights_only)
                break
            except Exception as exc:
                last_error = exc
        if checkpoint is None:
            raise last_error
        if isinstance(checkpoint, dict):
            checkpoint_meta = checkpoint
            # Prioritize EMA model if it exists, as it usually has better performance
            if 'ema_model' in checkpoint:
                state_dict = checkpoint['ema_model']
                print("Loaded EMA model weights.")
            else:
                state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        else:
            state_dict = checkpoint
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model_dict = model.state_dict()
        compatible_state = {
            key: value for key, value in state_dict.items()
            if key in model_dict and model_dict[key].shape == value.shape
        }
        model_dict.update(compatible_state)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: weights not found at {weights_path}")
    model.eval()
    return model, checkpoint_meta


def parse_weight_paths(weights_arg):
    if isinstance(weights_arg, (list, tuple)):
        return [str(path).strip() for path in weights_arg if str(path).strip()]
    return [path.strip() for path in str(weights_arg).split(',') if path.strip()]


def resolve_eval_image_size(args_img_size, checkpoint_meta):
    if args_img_size is not None:
        return args_img_size

    config_path = checkpoint_meta.get('config_path') if isinstance(checkpoint_meta, dict) else None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            image_size = config.get('training', {}).get('image_size')
            if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                return int(image_size[0])
            if isinstance(image_size, int):
                return image_size
        except Exception as exc:
            print(f"Warning: failed to read image_size from config {config_path}: {exc}")

    return 512


def decode_predictions(predictions, conf_threshold=0.5, iou_threshold=0.5):
    predictions = predictions.detach().cpu().numpy()
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.size == 0:
        return []
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    if class_scores.size == 0:
        return []
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    keep_mask = scores >= conf_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    detections = []
    for class_id in np.unique(class_ids):
        cls_mask = class_ids == class_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep_indices = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep_indices:
            detections.append({
                'bbox': cls_boxes[idx].tolist(),
                'score': float(cls_scores[idx]),
                'class_id': int(class_id)
            })
    detections.sort(key=lambda item: item['score'], reverse=True)
    return detections


def flip_detections_horizontally(detections, image_width):
    flipped = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        flipped.append({
            'bbox': [image_width - x2, y1, image_width - x1, y2],
            'score': det['score'],
            'class_id': det['class_id']
        })
    return flipped


def merge_detection_groups(detection_groups, iou_threshold=0.55):
    merged = []
    class_ids = sorted({
        det['class_id']
        for group in detection_groups
        for det in group
    })

    for class_id in class_ids:
        class_detections = [
            det for group in detection_groups for det in group
            if det['class_id'] == class_id
        ]
        if not class_detections:
            continue

        boxes = np.array([det['bbox'] for det in class_detections], dtype=np.float32)
        scores = np.array([det['score'] for det in class_detections], dtype=np.float32)
        order = np.argsort(scores)[::-1]

        while len(order) > 0:
            anchor = order[0]
            anchor_box = boxes[anchor]
            overlaps = np.array([box_iou(anchor_box, boxes[idx]) for idx in order])
            cluster_mask = overlaps >= iou_threshold
            cluster_indices = order[cluster_mask]
            cluster_scores = scores[cluster_indices]
            cluster_boxes = boxes[cluster_indices]

            weight_sum = float(cluster_scores.sum())
            if weight_sum > 0:
                fused_box = (cluster_boxes * cluster_scores[:, None]).sum(axis=0) / weight_sum
                fused_score = float(cluster_scores.mean())
            else:
                fused_box = cluster_boxes.mean(axis=0)
                fused_score = float(cluster_scores.max()) if len(cluster_scores) > 0 else 0.0

            merged.append({
                'bbox': fused_box.tolist(),
                'score': fused_score,
                'class_id': int(class_id)
            })
            order = order[~cluster_mask]

    merged.sort(key=lambda item: item['score'], reverse=True)
    return merged


def infer_sample_detections(models,
                            image_tensor,
                            conf_threshold,
                            iou_threshold,
                            use_tta=False,
                            ensemble_iou=0.55):
    detection_groups = []
    image_tensor = image_tensor.unsqueeze(0)
    image_width = int(image_tensor.shape[-1])

    for model in models:
        outputs = model(image_tensor)
        detections = decode_predictions(outputs[0], conf_threshold, iou_threshold)
        detection_groups.append(detections)

        if use_tta:
            flipped_images = torch.flip(image_tensor, dims=[3])
            flipped_outputs = model(flipped_images)
            flipped_detections = decode_predictions(flipped_outputs[0], conf_threshold, iou_threshold)
            detection_groups.append(
                flip_detections_horizontally(flipped_detections, image_width)
            )

    if len(detection_groups) == 1:
        return detection_groups[0]
    return merge_detection_groups(detection_groups, iou_threshold=ensemble_iou)


def evaluate_map(model,
                 dataloader,
                 device,
                 conf_threshold=0.5,
                 iou_threshold=0.5,
                 num_classes=1,
                 map_conf_threshold=None,
                 tta=False,
                 ensemble_iou=0.55):
    models = model if isinstance(model, (list, tuple)) else [model]
    for single_model in models:
        single_model.eval()
    all_predictions = []
    all_map_predictions = []
    all_targets = []
    if map_conf_threshold is None:
        map_conf_threshold = min(conf_threshold, 0.05)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            targets = batch['targets']
            for sample_idx in range(images.shape[0]):
                detections = infer_sample_detections(
                    models=models,
                    image_tensor=images[sample_idx],
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    use_tta=tta,
                    ensemble_iou=ensemble_iou
                )
                map_detections = infer_sample_detections(
                    models=models,
                    image_tensor=images[sample_idx],
                    conf_threshold=map_conf_threshold,
                    iou_threshold=iou_threshold,
                    use_tta=tta,
                    ensemble_iou=ensemble_iou
                )
                sample_targets = []
                target = targets[sample_idx]
                if len(target) > 0:
                    for t in target:
                        _, class_id, x1, y1, x2, y2 = t.tolist()
                        sample_targets.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': int(class_id)
                        })
                all_predictions.append(detections)
                all_map_predictions.append(map_detections)
                all_targets.append(sample_targets)
    map_results = calculate_map(all_map_predictions, all_targets, iou_thresholds=[iou_threshold], num_classes=num_classes)
    pr_results = calculate_precision_recall(all_predictions, all_targets, iou_threshold=iou_threshold)
    return (
        map_results.get(f'mAP@{iou_threshold}', 0.0),
        pr_results['precision'],
        pr_results['recall'],
        map_results.get(f'AP_per_class@{iou_threshold}', {})
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='outputs/exp_anchor_loss/weights/best.pt', help='path to model weights, comma separated for ensemble')
    parser.add_argument('--data', type=str, default=resolve_default_data_yaml(), help='path to dataset yaml')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--map-conf', type=float, default=0.05, help='confidence threshold used for mAP calculation')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS and metrics')
    parser.add_argument('--img-size', type=int, default=None, help='input image size, defaults to training image size in checkpoint')
    parser.add_argument('--batch-size', type=int, default=1, help='evaluation batch size')
    parser.add_argument('--tta', action='store_true', help='enable horizontal flip TTA')
    parser.add_argument('--ensemble-iou', type=float, default=0.55, help='IoU threshold for detection fusion when using ensemble/TTA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_config = load_data_config(args.data)
    num_classes = int(data_config['nc'])
    weight_paths = parse_weight_paths(args.weights)
    models = []
    checkpoint_metas = []
    for weight_path in weight_paths:
        built_model, checkpoint_meta = build_model(weight_path, device, num_classes)
        models.append(built_model)
        checkpoint_metas.append(checkpoint_meta)
    checkpoint_meta = checkpoint_metas[0] if checkpoint_metas else {}
    eval_img_size = resolve_eval_image_size(args.img_size, checkpoint_meta)

    val_split = 'val' if split_exists(args.data, data_config, 'val') else 'test'
    val_dataset = RoboflowFarmlandDataset(args.data, val_split, (eval_img_size, eval_img_size), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(
        f"\nEvaluating on {val_split} set ({len(val_dataset)} images), "
        f"img_size={eval_img_size}, models={len(models)}, tta={args.tta}..."
    )
    ap, precision, recall, per_class_ap = evaluate_map(
        models,
        val_loader,
        device,
        conf_threshold=args.conf,
        map_conf_threshold=args.map_conf,
        iou_threshold=args.iou,
        num_classes=num_classes,
        tta=args.tta,
        ensemble_iou=args.ensemble_iou
    )

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results (conf={args.conf}, map_conf={args.map_conf}, iou={args.iou})")
    print(f"{'=' * 50}")
    print(f"mAP@0.5: {ap:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if per_class_ap:
        print("Per-class AP:")
        for class_id, class_ap in per_class_ap.items():
            print(f"  class {class_id}: {class_ap:.4f}")
    print(f"{'=' * 50}")
