"""
极限训练：1301张极限增强农田数据 + yolov10m 大模型 + TTA评估
Windows兼容版本
"""

from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 获取Farmland_Crack_Detection根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Farmland_Crack_Detection目录
    
    data_yaml = os.path.join(project_root, 'data', 'data.yaml')
    data_dir = os.path.dirname(data_yaml)

    print(f"项目根目录: {project_root}")
    print(f"数据配置文件: {data_yaml}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"找不到数据配置文件: {data_yaml}")
    
    with open(data_yaml, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith('path:'):
            new_lines.append(f'path: {data_dir}\n')
        else:
            new_lines.append(line)
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"已更新data.yaml中的path为: {data_dir}")

    model = YOLO('yolov10m.pt')

    results = model.train(
        data=data_yaml,
        epochs=200,
        imgsz=1024,
        batch=4,
        device=0,
        lr0=0.001,
        lrf=0.01,
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        close_mosaic=40,
        mosaic=0.9,
        mixup=0.15,
        copy_paste=0.3,
        degrees=30.0,
        scale=0.6,
        shear=10.0,
        perspective=0.001,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.03,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.3,
        patience=40,
        project=os.path.join(project_root, 'runs'),
        name='extreme_m',
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    best_pt = os.path.join(project_root, 'runs', 'extreme_m', 'weights', 'best.pt')
    print(f'\n训练完成！模型: {best_pt}\n')

    print('=== TTA 评估 (test集) ===')
    model = YOLO(best_pt)
    test_results = model.val(
        data=data_yaml,
        split='test',
        imgsz=1024,
        batch=4,
        device=0,
        augment=True,
        conf=0.25,
        iou=0.5,
        project=os.path.join(project_root, 'runs'),
        name='extreme_m_tta_test'
    )
    print(f'TTA test mAP50: {test_results.box.map50:.4f}')

    print(f'\n=== TTA 评估 (val集) ===')
    val_results = model.val(
        data=data_yaml,
        split='val',
        imgsz=1024,
        batch=4,
        device=0,
        augment=True,
        conf=0.25,
        iou=0.5,
        project=os.path.join(project_root, 'runs'),
        name='extreme_m_tta_val'
    )
    print(f'TTA val mAP50: {val_results.box.map50:.4f}')