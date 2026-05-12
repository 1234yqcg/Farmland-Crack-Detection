# 功能：将训练集大图滑窗裁剪为 640x640 重叠 Patch，扩增样本数量
# 对于尺寸大于 640 的图像，以 320 步长滑窗裁剪，并自动转换 YOLO 标签坐标。
# 仅保留与目标重叠面积 >30% 的 Patch，减少截断框噪声。
# 输出到 data/train_patches/，并生成 dataset_patch.yaml 供训练使用。
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
import shutil
from PIL import Image

PATCH_SIZE = 640
STRIDE = 320
MIN_SIDE = 800
MIN_BOX_AREA_RATIO = 0.3


def resize_image_and_labels(image, labels, target_height):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height
    resized = cv2.resize(image, (new_w, new_h))
    # Labels are normalized in YOLO format, so they stay unchanged after uniform resize.
    return resized, labels


def extract_patches(image, labels, patch_size, stride):
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        if y + patch_size > h:
            y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            if x + patch_size > w:
                x = w - patch_size
            patch_img = image[y:y + patch_size, x:x + patch_size]
            patch_labels = []
            for label in labels:
                cls_id, cx, cy, bw, bh = label
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                ox1 = max(x1, x)
                oy1 = max(y1, y)
                ox2 = min(x2, x + patch_size)
                oy2 = min(y2, y + patch_size)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                orig_area = (x2 - x1) * (y2 - y1)
                overlap_area = (ox2 - ox1) * (oy2 - oy1)
                if overlap_area / orig_area < MIN_BOX_AREA_RATIO:
                    continue
                new_cx = ((ox1 + ox2) / 2 - x) / patch_size
                new_cy = ((oy1 + oy2) / 2 - y) / patch_size
                new_bw = (ox2 - ox1) / patch_size
                new_bh = (oy2 - oy1) / patch_size
                new_cx = max(0.0, min(1.0, new_cx))
                new_cy = max(0.0, min(1.0, new_cy))
                new_bw = max(0.0, min(1.0, new_bw))
                new_bh = max(0.0, min(1.0, new_bh))
                if new_bw < 0.01 or new_bh < 0.01:
                    continue
                patch_labels.append([cls_id, new_cx, new_cy, new_bw, new_bh])
            if patch_labels:
                patches.append((patch_img, patch_labels))
    return patches


def main():
    parser = argparse.ArgumentParser(description="Extract overlapping patches from training images")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to Farmland_Crack_Detection/data directory")
    parser.add_argument("--output-dir", type=str, default="data/train_patches",
                        help="Output directory for patches (relative to data-dir)")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--min-side", type=int, default=MIN_SIDE)
    parser.add_argument("--min-ratio", type=float, default=MIN_BOX_AREA_RATIO)
    args = parser.parse_args()

    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        base_dir = Path(__file__).resolve().parent.parent / "data"

    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"

    out_dir = base_dir / args.output_dir
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(train_img_dir.glob("*.jpg")) + sorted(train_img_dir.glob("*.png"))
    print(f"Found {len(image_files)} training images")

    total_patches = 0
    total_before = 0
    total_after = 0

    for img_path in tqdm(image_files, desc="Extracting patches"):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        try:
            image = Image.open(str(img_path))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        except Exception:
            continue

        labels = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    labels.append([cls_id, cx, cy, bw, bh])

        if not labels:
            continue

        total_before += len(labels)

        resized_img, resized_labels = resize_image_and_labels(image, labels, args.min_side)

        patches = extract_patches(resized_img, resized_labels, args.patch_size, args.stride)

        for patch_idx, (patch_img, patch_labels) in enumerate(patches):
            patch_name = f"{img_path.stem}_p{patch_idx:02d}"
            patch_img_path = out_img_dir / f"{patch_name}.jpg"
            Image.fromarray(patch_img).save(str(patch_img_path), quality=95)

            patch_lbl_path = out_lbl_dir / f"{patch_name}.txt"
            with open(patch_lbl_path, "w") as f:
                for pl in patch_labels:
                    cls_id, cx, cy, bw, bh = pl
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            total_patches += 1
            total_after += len(patch_labels)

    print(f"\n{'='*50}")
    print(f"Patch extraction complete!")
    print(f"  Original images: {len(image_files)}")
    print(f"  Output patches: {total_patches}")
    print(f"  Original boxes: {total_before}")
    print(f"  Patch boxes: {total_after}")
    print(f"  Avg patches per image: {total_patches / max(len(image_files), 1):.1f}")
    print(f"  Avg boxes per patch: {total_after / max(total_patches, 1):.1f}")
    print(f"  Output: {out_dir}")
    print(f"{'='*50}")

    dataset_yaml = {
        'train': str(out_dir),
        'val': str(base_dir / 'val'),
        'test': str(base_dir / 'test'),
        'nc': 3,
        'names': ['细微裂纹', '网状裂隙', '深大裂缝']
    }
    yaml_path = out_dir.parent / 'dataset_patch.yaml'
    with open(str(yaml_path), 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, default_flow_style=False)
    print(f"Dataset config saved: {yaml_path}")


if __name__ == "__main__":
    main()
