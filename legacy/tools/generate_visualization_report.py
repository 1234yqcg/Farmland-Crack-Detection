import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import Counter, defaultdict
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

# 设置matplotlib后端为非交互式（Linux服务器必需）
matplotlib.use('Agg')

# Linux兼容的字体设置（优先使用系统字体）
def setup_matplotlib_font():
    """设置matplotlib字体，兼容Linux和Windows"""
    import platform
    system = platform.system()
    
    if system == 'Linux':
        # Linux系统字体优先级
        font_list = ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'sans-serif']
        # 尝试查找中文字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
        for cf in chinese_fonts:
            try:
                from matplotlib import font_manager
                fonts = [f.name for f in font_manager.fontManager.ttflist]
                if cf in fonts:
                    font_list.insert(0, cf)
                    break
            except:
                pass
        
        matplotlib.rcParams['font.sans-serif'] = font_list
    else:
        # Windows/Mac
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
    matplotlib.rcParams['axes.unicode_minus'] = False

setup_matplotlib_font()


def setup_style():
    """设置图表样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['font.size'] = 11


def read_tensorboard_events(event_file):
    """读取TensorBoard日志"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(event_file)
        ea.Reload()
        data = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        return data
    except Exception as e:
        print(f"[WARN] Cannot read TensorBoard: {e}")
        return {}


def generate_results_png(tb_data, output_dir):
    """
    生成训练结果图 (results.png)
    包含：Loss曲线、mAP/Precision/Recall/F1曲线、学习率曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss曲线
    loss_tags = {
        'loss/train': ('Train Loss', 'steelblue'),
        'loss/val': ('Val Loss', 'coral'),
    }
    ax = axes[0][0]
    for tag, (label, color) in loss_tags.items():
        if tag in tb_data and len(tb_data[tag]) > 0:
            steps, values = zip(*tb_data[tag])
            ax.plot(steps, values, label=label, color=color, linewidth=2)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 指标曲线 (mAP, Precision, Recall, F1)
    metric_tags = [
        ('metrics/mAP50', 'mAP@0.5', '#2ecc71'),
        ('metrics/precision', 'Precision', '#3498db'),
        ('metrics/recall', 'Recall', '#e74c3c'),
        ('metrics/f1', 'F1-Score', '#9b59b6'),
    ]
    ax = axes[0][1]
    for tag, label, color in metric_tags:
        if tag in tb_data and len(tb_data[tag]) > 0:
            steps, values = zip(*tb_data[tag])
            ax.plot(steps, values, marker='o', markersize=4, label=label,
                    color=color, linewidth=2)
    ax.set_title('Metrics Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 学习率曲线
    lr_tags = [tag for tag in tb_data.keys() if 'lr' in tag.lower()]
    ax = axes[1][0]
    if lr_tags:
        tag = lr_tags[0]
        if tag in tb_data and len(tb_data[tag]) > 0:
            steps, values = zip(*tb_data[tag])
            ax.plot(steps, values, color='#9b59b6', linewidth=2)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)

    # 训练信息面板
    ax = axes[1][1]
    ax.axis('off')
    
    # 提取最佳指标
    best_map = 0
    best_epoch = 0
    if 'metrics/mAP50' in tb_data and len(tb_data['metrics/mAP50']) > 0:
        for step, val in tb_data['metrics/mAP50']:
            if val > best_map:
                best_map = val
                best_epoch = step
    
    info_text = (
        f"Training Results Summary\n\n"
        f"Best mAP@0.5: {best_map:.4f} (Epoch {best_epoch})\n"
        f"\n"
        f"- Top-Left: Train/Val Loss Curves\n"
        f"- Top-Right: mAP/Precision/Recall/F1\n"
        f"- Bottom-Left: Learning Rate Schedule\n"
        f"- Bottom-Right: Training Summary"
    )
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    plt.suptitle('YOLOv10 Farmland Crack Detection - Training Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'results.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: results.png")
    return path


def generate_confusion_matrix_png(model_path, data_yaml, output_dir, device='cuda'):
    """
    生成混淆矩阵图 (confusion_matrix.png)
    """
    from utils.roboflow_dataset import RoboflowFarmlandDataset

    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    names = cfg.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    num_classes = len(names)

    test_dataset = RoboflowFarmlandDataset(
        data_yaml_path=data_yaml,
        split='test',
        image_size=(800, 800),
        transform=None,
        augment=False
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model', checkpoint)

    from models.yolov10_crack import YOLOv10Crack
    model = YOLOv10Crack(num_classes=num_classes).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            img_tensor = sample['image'].unsqueeze(0).to(device)

            outputs = model(img_tensor)[0]
            pred_cls_ids = []
            if outputs is not None:
                for det in outputs:
                    det_list = [x.cpu().float() if hasattr(x, 'cpu') else float(x) for x in det]
                    obj_conf = float(det_list[4])
                    cls_scores = [float(x) for x in det_list[5:]]
                    cls_id = int(np.argmax(cls_scores))
                    cls_conf = float(max(cls_scores))
                    if obj_conf * cls_conf > 0.25:
                        pred_cls_ids.append(cls_id)

            gt_labels = sample['labels']
            gt_cls_ids = []
            if gt_labels.numel() > 0:
                for t in gt_labels:
                    parts = [x.item() if hasattr(x, 'item') else x for x in t]
                    if len(parts) >= 6:
                        gt_cls_ids.append(int(parts[1]))
                    elif len(parts) >= 5:
                        gt_cls_ids.append(int(parts[1]))

            for gc in gt_cls_ids:
                pc = pred_cls_ids[0] if pred_cls_ids else num_classes
                conf_matrix[gc, pc] += 1

            for j in range(len(pred_cls_ids), max(len(gt_cls_ids), len(pred_cls_ids))):
                if j >= len(gt_cls_ids):
                    conf_matrix[num_classes, pred_cls_ids[j]] += 1

    class_names_bg = names + ['Background']

    fig, ax = plt.subplots(figsize=(10, 8))
    conf_matrix_norm = conf_matrix.astype(float) / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)

    im = ax.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=1)

    thresh = conf_matrix_norm.max() / 2.
    for i in range(len(class_names_bg)):
        for j in range(len(class_names_bg)):
            text_color = 'white' if conf_matrix_norm[i, j] > thresh else 'black'
            val_str = str(int(conf_matrix[i, j]))
            pct_str = f'{conf_matrix_norm[i,j]:.1%}'
            ax.text(j, i, f'{val_str}\n({pct_str})',
                   ha='center', va='center', color=text_color, fontsize=9)

    ax.set_xticks(range(len(class_names_bg)))
    ax.set_yticks(range(len(class_names_bg)))
    ax.set_xticklabels(class_names_bg, rotation=45, ha='right')
    ax.set_yticklabels(class_names_bg)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix | Test Set | conf=0.25',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Normalized Ratio', rotation=-90, va='bottom')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: confusion_matrix.png")
    return path


def generate_pr_curve_png(model_path, data_yaml, output_dir, device='cuda'):
    """
    生成PR曲线图 (PR_curve.png)
    """
    from utils.roboflow_dataset import RoboflowFarmlandDataset

    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    names = cfg.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    num_classes = len(names)

    test_dataset = RoboflowFarmlandDataset(
        data_yaml_path=data_yaml,
        split='test',
        image_size=(800, 800),
        transform=None,
        augment=False
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model', checkpoint)

    from models.yolov10_crack import YOLOv10Crack
    model = YOLOv10Crack(num_classes=num_classes).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    all_precisions = defaultdict(list)
    all_recalls = defaultdict(list)
    class_gt_counts = Counter()

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            img_tensor = sample['image'].unsqueeze(0).to(device)

            outputs = model(img_tensor)[0]
            preds = []
            if outputs is not None:
                for det in outputs:
                    det_cpu = [x.cpu() if hasattr(x, 'cpu') else x for x in det]
                    cx, cy, w, h, obj_conf, *cls_scores = det_cpu
                    cls_id = int(np.argmax(cls_scores))
                    cls_conf = float(max(cls_scores))
                    final_conf = float(obj_conf) * cls_conf
                    x1 = cx - w / 2; y1 = cy - h / 2
                    x2 = cx + w / 2; y2 = cy + h / 2
                    preds.append((cls_id, final_conf, [float(x1), float(y1), float(x2), float(y2)]))

            gt_labels = sample['labels']
            gt_boxes = []
            if gt_labels.numel() > 0:
                for t in gt_labels:
                    if len(t) >= 6:
                        _, cls_id, cx, cy, w, h = t
                    else:
                        _, cls_id, cx, cy, w, h = t
                    gt_boxes.append({
                        'class_id': int(cls_id.item()),
                        'bbox': [float((cx - w/2)), float((cy - h/2)), float((cx + w/2)), float((cy + h/2))]
                    })
                    class_gt_counts[int(cls_id.item())] += 1

            for cls_id in range(num_classes):
                cls_preds = sorted([p for p in preds if p[0] == cls_id], key=lambda x: -x[1])
                cls_gts = [g for g in gt_boxes if g['class_id'] == cls_id]

                matched_gts = set()
                tp_cumsum = 0
                fp_cumsum = 0

                for pred in cls_preds:
                    iou_max = 0
                    best_gt_idx = -1
                    for gi, gt in enumerate(cls_gts):
                        if gi in matched_gts:
                            continue
                        pb = pred[2]; gb = gt['bbox']
                        ix1 = max(pb[0], gb[0]); iy1 = max(pb[1], gb[1])
                        ix2 = min(pb[2], gb[2]); iy2 = min(pb[3], gb[3])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        area_p = (pb[2]-pb[0])*(pb[3]-pb[1])
                        area_g = (gb[2]-gb[0])*(gb[3]-gb[1])
                        union = area_p + area_g - inter
                        iou = inter / union if union > 0 else 0
                        if iou > iou_max:
                            iou_max = iou; best_gt_idx = gi

                    if iou_max > 0.5 and best_gt_idx >= 0:
                        tp_cumsum += 1
                        matched_gts.add(best_gt_idx)
                    else:
                        fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
                recall = tp_cumsum / class_gt_counts[cls_id] if class_gt_counts[cls_id] > 0 else 0
                all_precisions[cls_id].append(precision)
                all_recalls[cls_id].append(recall)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_id in range(num_classes):
        precisions = all_precisions[cls_id]
        recalls = all_recalls[cls_id]
        if len(precisions) == 0:
            continue

        indices = np.argsort(recalls)
        sorted_r = [recalls[i] for i in indices]
        sorted_p = [precisions[i] for i in indices]

        sorted_r = [0] + list(sorted_r) + [sorted_r[-1] if sorted_r else 1]
        sorted_p = [1] + list(sorted_p) + [0]

        ap = np.trapz(sorted_p, sorted_r)
        ax.plot(sorted_r, sorted_p, color=colors[cls_id % len(colors)],
                linewidth=2, label=f'{names[cls_id]} (AP={ap:.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('PR Curve (Precision-Recall Curve) | Test Set | IoU=0.5',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'PR_curve.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: PR_curve.png")
    return path


def generate_f1_curve_png(tb_data, output_dir):
    """
    生成F1曲线图 (F1_curve.png)
    基于TensorBoard中的F1数据绘制
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 尝试从TensorBoard读取F1数据
    f1_tag = 'metrics/f1'
    if f1_tag in tb_data and len(tb_data[f1_tag]) > 0:
        steps, values = zip(*tb_data[f1_tag])
        ax.plot(steps, values, color='#9b59b6', linewidth=2, marker='o', markersize=4, label='F1-Score')
        
        # 标注最佳点
        max_idx = np.argmax(values)
        ax.annotate(f'Best: {values[max_idx]:.4f}\n(Epoch {steps[max_idx]})',
                   xy=(steps[max_idx], values[max_idx]),
                   xytext=(steps[max_idx]+len(steps)*0.1, values[max_idx]),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')
        
        # 添加参考线
        ax.axhline(y=np.mean(values), color='#3498db', linestyle='--', linewidth=1.5,
                  label=f'Mean: {np.mean(values):.4f}')
    else:
        # 如果没有F1数据，显示提示
        ax.text(0.5, 0.5, 'No F1-Score data found in TensorBoard logs',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Curve During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'F1_curve.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: F1_curve.png")
    return path


def generate_labels_jpg(data_yaml, output_dir, num_samples=6):
    """
    生成标签可视化图 (labels.jpg)
    显示数据集中的标注样本
    """
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    names = cfg.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    base_dir = Path(cfg['path'])

    # 尝试获取test集，如果没有则使用val集或train集
    test_split = None
    for split in ['test', 'val', 'train']:
        if split in cfg:
            test_split = split
            break
    
    if not test_split:
        print("[WARN] No valid dataset split found")
        return None

    test_img_dir = base_dir / cfg[test_split].replace('/images', '/images')
    test_lbl_dir = base_dir / cfg[test_split].replace('/images', '/labels')

    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_files.extend(test_img_dir.glob(ext))
    img_files = sorted(img_files)[:num_samples]

    if not img_files:
        print("[WARN] No images found")
        return None

    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, img_path in enumerate(img_files):
        if idx >= len(axes):
            break

        ax = axes[idx]
        img = Image.open(img_path).convert('RGB')

        lbl_path = test_lbl_dir / (img_path.stem + '.txt')
        boxes = []
        if lbl_path.exists():
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        W, H = img.size
                        x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
                        x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
                        boxes.append((x1, y1, x2, y2, cls_id))

        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2, cls_id in boxes:
            color = colors[cls_id % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制类别标签
            label_text = names[cls_id] if cls_id < len(names) else f'class_{cls_id}'
            draw.text((x1+2, y1-12), label_text, fill=color)

        ax.imshow(np.array(img))
        ax.set_title(f'{img_path.name}\n({len(boxes)} annotations)', fontsize=10)
        ax.axis('off')

    for idx in range(len(img_files), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Dataset Labels Visualization ({test_split} set)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # 保存为JPG格式
    path = os.path.join(output_dir, 'labels.jpg')
    plt.savefig(path, bbox_inches='tight', facecolor='white', format='jpeg', quality=95)
    plt.close()
    print(f"  [OK] Saved: labels.jpg")
    return path


def generate_all_visualizations(output_dir, model_path, data_yaml, config_path, tb_log_dir):
    """
    生成所有可视化报告的主函数
    
    Args:
        output_dir: 训练输出目录
        model_path: 最佳模型路径
        data_yaml: 数据集配置文件路径
        config_path: 训练配置文件路径
        tb_log_dir: TensorBoard日志目录
    """
    setup_style()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("YOLOv10 Farmland Crack Detection - Visualization Report Generator")
    print("=" * 60)
    print(f"Output dir: {output_dir}")
    print()

    # 读取TensorBoard日志
    tb_data = {}
    tb_files = list(Path(tb_log_dir).glob('events.out.tfevents.*')) if tb_log_dir else []
    
    if tb_files:
        print("[1/5] Reading training logs...")
        tb_data = read_tensorboard_events(str(tb_files[0]))
        print(f"  Found {len(tb_data)} metrics")
    else:
        print("[WARN] No TensorBoard logs found, some plots will be skipped")

    # 生成results.png
    print("\n[2/5] Generating results.png...")
    try:
        if tb_data:
            generate_results_png(tb_data, output_dir)
        else:
            print("  [SKIP] No training log data available")
    except Exception as e:
        print(f"  [WARN] Failed to generate results.png: {e}")

    # 生成confusion_matrix.png
    print("\n[3/5] Generating confusion_matrix.png...")
    try:
        if os.path.exists(model_path):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            generate_confusion_matrix_png(model_path, data_yaml, output_dir, device)
        else:
            print(f"  [SKIP] Model file not found: {model_path}")
    except Exception as e:
        print(f"  [WARN] Failed to generate confusion_matrix.png: {e}")

    # 生成PR_curve.png
    print("\n[4/5] Generating PR_curve.png...")
    try:
        if os.path.exists(model_path):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            generate_pr_curve_png(model_path, data_yaml, output_dir, device)
        else:
            print(f"  [SKIP] Model file not found: {model_path}")
    except Exception as e:
        print(f"  [WARN] Failed to generate PR_curve.png: {e}")

    # 生成F1_curve.png
    print("\n[5/5] Generating F1_curve.png...")
    try:
        if tb_data:
            generate_f1_curve_png(tb_data, output_dir)
        else:
            print("  [SKIP] No F1 data available")
    except Exception as e:
        print(f"  [WARN] Failed to generate F1_curve.png: {e}")

    # 生成labels.jpg
    print("\n[BONUS] Generating labels.jpg...")
    try:
        generate_labels_jpg(data_yaml, output_dir)
    except Exception as e:
        print(f"  [WARN] Failed to generate labels.jpg: {e}")

    print()
    print("=" * 60)
    print("  Visualization report generation complete!")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'outputs', 'visualization_report')
    
    model_path = os.path.join(project_root, 'outputs', 'farmland_finetune_v15', 'weights', 'best.pt')
    data_yaml = os.path.join(project_root, 'data', 'dataset_v15_optimized.yaml')
    config_path = os.path.join(project_root, 'configs', 'train_farmland_v15.yaml')
    tb_log_dir = os.path.join(project_root, 'outputs', 'farmland_finetune_v15', 'logs')
    
    generate_all_visualizations(
        output_dir=output_dir,
        model_path=model_path,
        data_yaml=data_yaml,
        config_path=config_path,
        tb_log_dir=tb_log_dir
    )
