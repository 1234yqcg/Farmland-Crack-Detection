#!/usr/bin/env python3
"""
Data Optimization Pipeline for v16 (Three-Class)
- 保持三分类标签（细微/网状/深大）
- 过滤负样本（~150张）
- 图像切片（640×640）增强小目标
"""

import os
import sys
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, 'data')

    src_train_img = os.path.join(data_root, 'dataset_balanced', 'train', 'images')
    src_train_lbl = os.path.join(data_root, 'dataset_balanced', 'train', 'labels')
    src_val_img = os.path.join(data_root, 'dataset_balanced', 'val', 'images')
    src_val_lbl = os.path.join(data_root, 'dataset_balanced', 'val', 'labels')
    src_test_img = os.path.join(data_root, 'dataset_balanced', 'test', 'images')
    src_test_lbl = os.path.join(data_root, 'dataset_balanced', 'test', 'labels')

    out_dir = os.path.join(data_root, 'dataset_v16_optimized')
    out_train_img = os.path.join(out_dir, 'train', 'images')
    out_train_lbl = os.path.join(out_dir, 'train', 'labels')
    out_val_img = os.path.join(out_dir, 'val', 'images')
    out_val_lbl = os.path.join(out_dir, 'val', 'labels')
    out_test_img = os.path.join(out_dir, 'test', 'images')
    out_test_lbl = os.path.join(out_dir, 'test', 'labels')

    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl, out_test_img, out_test_lbl]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Copy all data (keep original 3-class labels)")
    print("=" * 60)

    def process_split(src_img_d, src_lbl_d, dst_img_d, dst_lbl_d, split_name):
        all_imgs = list(Path(src_img_d).glob('*.jpg')) + list(Path(src_img_d).glob('*.jpeg'))
        labeled_count = 0
        box_count = 0

        for img_path in tqdm(all_imgs, desc=f"  {split_name}"):
            stem = img_path.stem
            shutil.copy2(img_path, Path(dst_img_d) / img_path.name)

            lbl_path = Path(src_lbl_d) / (stem + '.txt')
            if lbl_path.exists():
                shutil.copy2(lbl_path, Path(dst_lbl_d) / (stem + '.txt'))
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            box_count += 1
                labeled_count += 1

        unlabeled = len(all_imgs) - labeled_count
        print(f"  {split_name}: {len(all_imgs)} images ({labeled_count} labeled, {unlabeled} unlabeled), {box_count} boxes")
        return len(all_imgs), labeled_count, unlabeled, box_count

    t_total, t_labeled, t_unlabeled, t_boxes = process_split(
        src_train_img, src_train_lbl, out_train_img, out_train_lbl, "Train")
    v_total, v_labeled, v_unlabeled, v_boxes = process_split(
        src_val_img, src_val_lbl, out_val_img, out_val_lbl, "Val")
    te_total, te_labeled, te_unlabeled, te_boxes = process_split(
        src_test_img, src_test_lbl, out_test_img, out_test_lbl, "Test")

    print("\n" + "=" * 60)
    print("Phase 2: Filter negative samples (keep ~150)")
    print("=" * 60)

    all_train_stems = [p.stem for p in Path(out_train_img).glob('*.jpg')] + [p.stem for p in Path(out_train_img).glob('*.jpeg')]
    labeled_stems = set(p.stem for p in Path(out_train_lbl).glob('*.txt'))
    unlabeled_stems = [s for s in all_train_stems if s not in labeled_stems]

    print(f"  Labeled images:   {len(labeled_stems)}")
    print(f"  Unlabeled images: {len(unlabeled_stems)}")

    max_negatives = 150
    if len(unlabeled_stems) > max_negatives:
        keep_negatives = set(random.sample(unlabeled_stems, max_negatives))
        remove_stems = [s for s in unlabeled_stems if s not in keep_negatives]

        for stem in remove_stems:
            for ext in ['.jpg', '.jpeg']:
                img_p = Path(out_train_img) / (stem + ext)
                if img_p.exists():
                    img_p.unlink()

        print(f"  Kept:  {max_negatives} negatives")
        print(f"  Removed: {len(remove_stems)} negatives")
    else:
        print(f"  All {len(unlabeled_stems)} kept (under limit)")

    remaining_imgs = len([p for p in Path(out_train_img).glob('*.jpg')]) + len([p for p in Path(out_train_img).glob('*.jpeg')])
    print(f"  Final train images: {remaining_imgs}")

    print("\n" + "=" * 60)
    print("Phase 3: Tile large images (640x640)")
    print("=" * 60)

    tile_size = 640
    overlap = 80
    stride = tile_size - overlap
    tile_count = 0
    tile_box_count = 0

    train_imgs = list(Path(out_train_img).glob('*.jpg')) + list(Path(out_train_img).glob('*.jpeg'))

    for img_path in tqdm(train_imgs, desc="  Tiling"):
        try:
            img = Image.open(img_path).convert('RGB')
            W, H = img.size

            if W <= tile_size and H <= tile_size:
                continue

            stem = img_path.stem
            lbl_path = Path(out_train_lbl) / (stem + '.txt')

            boxes = []
            if lbl_path.exists():
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
                            x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
                            boxes.append((cls_id, x1, y1, x2, y2))

            nx = max(1, int(np.ceil((W - tile_size) / stride)) + 1)
            ny = max(1, int(np.ceil((H - tile_size) / stride)) + 1)

            tile_idx = 0
            for iy in range(ny):
                for ix in range(nx):
                    tx = min(ix * stride, W - tile_size)
                    ty = min(iy * stride, H - tile_size)
                    tx = max(0, min(tx, W - tile_size))
                    ty = max(0, min(ty, H - tile_size))

                    tile_img = img.crop((tx, ty, tx + tile_size, ty + tile_size))

                    tile_boxes = []
                    for cls_id, bx1, by1, bx2, by2 in boxes:
                        ix1 = max(bx1 - tx, 0); iy1 = max(by1 - ty, 0)
                        ix2 = min(bx2 - tx, tile_size); iy2 = min(by2 - ty, tile_size)
                        if ix2 > ix1 and iy2 > iy1:
                            tcx = (ix1+ix2)/2/tile_size; tcy = (iy1+iy2)/2/tile_size
                            tw = (ix2-ix1)/tile_size; th = (iy2-iy1)/tile_size
                            if 0 < tw <= 1 and 0 < th <= 1:
                                tile_boxes.append(f"{cls_id} {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}\n")

                    if tile_boxes:
                        tname = f"{stem}_t{tile_idx}"
                        tile_img.save(Path(out_train_img) / (tname + '.jpg'), quality=95)
                        with open(Path(out_train_lbl) / (tname + '.txt'), 'w') as f:
                            f.writelines(tile_boxes)
                        tile_count += 1
                        tile_box_count += len(tile_boxes)
                    tile_idx += 1
        except Exception as e:
            print(f"  [WARN] Failed to tile {img_path.name}: {e}")
            continue

    print(f"  Tiles created: {tile_count}")
    print(f"  Boxes in tiles: {tile_box_count}")

    yaml_content = """path: ./data/dataset_v16_optimized
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: 细微裂纹
  1: 网状裂隙
  2: 深大裂缝
"""
    yaml_path = os.path.join(data_root, 'dataset_v16_optimized.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    for split in ['train', 'val', 'test']:
        idir = os.path.join(out_dir, split, 'images')
        ldir = os.path.join(out_dir, split, 'labels')
        
        imgs = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            imgs.extend(Path(idir).glob(ext))
        ni = len(imgs)
        
        lbls = list(Path(ldir).glob('*.txt'))
        nl = len(lbls)
        
        tb = 0
        for lf in lbls:
            try:
                with open(lf, 'r') as f:
                    tb += sum(1 for l in f if l.strip())
            except:
                pass
        
        print(f"  {split:>5}: {ni:>5} imgs, {nl:>4} lbls, {tb:>5} boxes ({tb/max(ni,1):.2f}/img)")

    print(f"\n  YAML: data/dataset_v16_optimized.yaml")
    print(f"  Dir:  {out_dir}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
