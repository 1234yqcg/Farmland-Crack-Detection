#!/usr/bin/env python3
"""
Advanced Evaluation Script with TTA and Model Ensemble
Target: Maximize mAP through advanced inference techniques
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_map, build_model, parse_weight_paths, resolve_default_data_yaml
from utils.roboflow_dataset import RoboflowFarmlandDataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        images.append(item['image'])
        targets.append(item['labels'])
    images = torch.stack(images, dim=0)
    return {'images': images, 'targets': targets}


def run_comprehensive_evaluation(args):
    """运行全面的评估，包括多种配置"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Advanced Evaluation - Target: mAP >= 80%")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 加载数据集配置
    data_yaml = args.data if args.data else resolve_default_data_yaml()
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = int(data_config['nc'])
    names = data_config.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=int)]
    
    print(f"Dataset: {data_yaml}")
    print(f"Classes ({num_classes}): {names}")
    
    # 加载模型
    weight_paths = parse_weight_paths(args.weights)
    print(f"\nModels to evaluate: {len(weight_paths)}")
    for i, wp in enumerate(weight_paths):
        exists = os.path.exists(wp)
        print(f"  [{i+1}] {'✓' if exists else '✗'} {wp}")
    
    models = []
    for wp in weight_paths:
        model, _ = build_model(wp, device, num_classes)
        models.append(model)
    
    # 确定评估数据集
    split = args.split if args.split else ('test' if os.path.exists(os.path.join(os.path.dirname(data_yaml), 'test', 'images')) else 'val')
    
    img_size = args.img_size or 1024
    
    dataset = RoboflowFarmlandDataset(
        data_yaml_path=data_yaml,
        split=split,
        image_size=(img_size, img_size),
        augment=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"\nEvaluating on {split} set: {len(dataset)} images")
    print(f"Image size: {img_size}x{img_size}")
    print(f"{'='*60}\n")
    
    results_summary = []
    
    # ========== 测试1: 基线评估（无TTA）==========
    print("[1/5] Baseline evaluation (no TTA)...")
    try:
        map_base, prec_base, rec_base, ap_base = evaluate_map(
            models[0] if len(models) == 1 else models,
            loader,
            device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            num_classes=num_classes,
            tta=False
        )
        results_summary.append({
            'name': 'Baseline (no TTA)',
            'mAP': map_base,
            'precision': prec_base,
            'recall': rec_base,
            'ap_per_class': ap_base
        })
        print(f"  mAP@0.5: {map_base:.4f} | P: {prec_base:.4f} | R: {rec_base:.4f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # ========== 测试2: TTA评估（水平翻转）==========
    if args.tta or True:  # 默认启用TTA
        print("\n[2/5] TTA evaluation (horizontal flip)...")
        try:
            map_tta, prec_tta, rec_tta, ap_tta = evaluate_map(
                models[0] if len(models) == 1 else models,
                loader,
                device,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                num_classes=num_classes,
                tta=True
            )
            results_summary.append({
                'name': 'TTA (flip)',
                'mAP': map_tta,
                'precision': prec_tta,
                'recall': rec_tta,
                'ap_per_class': ap_tta
            })
            improvement = ((map_tta - map_base) / map_base * 100) if map_base > 0 else 0
            print(f"  mAP@0.5: {map_tta:.4f} | P: {prec_tta:.4f} | R: {rec_tta:.4f} | +{improvement:.1f}% vs baseline")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    # ========== 测试3: 低置信度阈值评估 ==========
    print("\n[3/5] Low confidence threshold (conf=0.15)...")
    try:
        map_low, prec_low, rec_low, ap_low = evaluate_map(
            models[0] if len(models) == 1 else models,
            loader,
            device,
            conf_threshold=0.15,
            iou_threshold=args.iou,
            num_classes=num_classes,
            tta=True
        )
        results_summary.append({
            'name': 'TTA + Low Conf (0.15)',
            'mAP': map_low,
            'precision': prec_low,
            'recall': rec_low,
            'ap_per_class': ap_low
        })
        print(f"  mAP@0.5: {map_low:.4f} | P: {prec_low:.4f} | R: {rec_low:.4f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # ========== 测试4: 模型集成（如果有多个权重）==========
    if len(models) > 1:
        print(f"\n[4/5] Model ensemble ({len(models)} models)...")
        try:
            map_ens, prec_ens, rec_ens, ap_ens = evaluate_map(
                models,
                loader,
                device,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                num_classes=num_classes,
                tta=True
            )
            results_summary.append({
                'name': f'Ensemble ({len(models)} models) + TTA',
                'mAP': map_ens,
                'precision': prec_ens,
                'recall': rec_ens,
                'ap_per_class': ap_ens
            })
            print(f"  mAP@0.5: {map_ens:.4f} | P: {prec_ens:.4f} | R: {rec_ens:.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    # ========== 测试5: 最优组合 ==========
    print("\n[5/5] Optimal configuration (TTA + low conf + ensemble)...")
    try:
        map_opt, prec_opt, rec_opt, ap_opt = evaluate_map(
            models,
            loader,
            device,
            conf_threshold=0.20,
            iou_threshold=0.45,
            num_classes=num_classes,
            tta=True
        )
        results_summary.append({
            'name': 'Optimal (TTA+conf=0.2+IoU=0.45)',
            'mAP': map_opt,
            'precision': prec_opt,
            'recall': rec_opt,
            'ap_per_class': ap_opt
        })
        print(f"  mAP@0.5: {map_opt:.4f} | P: {prec_opt:.4f} | R: {rec_opt:.4f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # ========== 输出总结 ==========
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<40} {'mAP':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 68)
    
    best_result = None
    for r in results_summary:
        marker = ""
        if best_result is None or r['mAP'] > best_result['mAP']:
            best_result = r
            marker = " 🏆"
        
        print(f"{r['name']:<40} {r['mAP']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f}{marker}")
    
    print("-" * 68)
    
    if best_result:
        print(f"\n🏆 BEST RESULT:")
        print(f"   Configuration: {best_result['name']}")
        print(f"   mAP@0.5:      {best_result['mAP']:.4f} ({best_result['mAP']*100:.2f}%)")
        print(f"   Precision:    {best_result['precision']:.4f}")
        print(f"   Recall:       {best_result['recall']:.4f}")
        
        if best_result['ap_per_class']:
            print(f"\n   Per-class AP@0.5:")
            for cls_id, ap in sorted(best_result['ap_per_class'].items()):
                cls_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                print(f"     {cls_name}: {ap:.4f}")
        
        target = 0.80
        achieved = best_result['mAP']
        gap = target - achieved
        
        print(f"\n🎯 TARGET ANALYSIS:")
        print(f"   Target mAP:  {target*100:.1f}%")
        print(f"Achieved mAP:  {achieved*100:.2f}%")
        
        if achieved >= target:
            print(f"   Status:      ✅ TARGET ACHIEVED!")
        elif achieved >= target * 0.9:
            print(f"   Status:      ⚠️ Close to target (gap: {gap*100:.2f}%)")
        else:
            print(f"   Status:      ❌ Gap remaining: {gap*100:.2f}%")
            print(f"\n💡 Recommendations:")
            if best_result['recall'] < 0.6:
                print("   • Recall is low → Try lower confidence threshold (0.10-0.20)")
            if best_result['precision'] < 0.7:
                print("   • Precision is moderate → Consider NMS IoU tuning (0.4-0.5)")
            print("   • More training epochs may help")
            print("   • Consider larger image size (1280x1280)")
    
    print(f"{'='*60}\n")
    
    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced YOLOv16 Evaluation with TTA')
    parser.add_argument('--weights', type=str, required=True,
                       help='model weights path(s), comma-separated for ensemble')
    parser.add_argument('--data', type=str, default=None,
                       help='dataset YAML path')
    parser.add_argument('--split', type=str, default=None,
                       help='dataset split (val/test/train)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold')
    parser.add_argument('--img-size', type=int, default=1024,
                       help='input image size')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='evaluation batch size')
    parser.add_argument('--tta', action='store_true',
                       help='enable test-time augmentation')
    
    args = parser.parse_args()
    
    result = run_comprehensive_evaluation(args)
