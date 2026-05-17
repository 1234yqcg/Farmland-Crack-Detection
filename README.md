# YOLOv10 农田裂缝检测系统

## 📋 项目简介

基于 **YOLOv10m** 的农田土壤裂缝智能检测系统，采用**极限数据增强策略**，实现三分类检测：细微裂纹(tiny_crack)、网状裂隙(mesh_crack)、深大裂缝(deep_crack)。

**当前版本**: v16 (极限增强版)

### 核心特性

- ✅ **三分类检测**: 细微裂纹 / 网状裂隙 / 深大裂缝
- ✅ **极限数据增强**: 1301张高质量增强样本，包含Copy-Paste、强色彩/光照变换、Mosaic+MixUp组合
- ✅ **大模型基础**: YOLOv10m (Medium)，平衡精度与速度
- ✅ **强鲁棒性**: 30°旋转、0.6缩放、HSV大幅扰动、随机擦除
- ✅ **自动TTA评估**: 训练结束后自动进行测试时增强评估
- ✅ **早停机制**: patience=40，避免过拟合
- ✅ **完整可视化**: 自动生成混淆矩阵、PR曲线、F1曲线等报告

---

## 🚀 快速开始

### 1. 环境要求

```bash
# Python 3.8+
python --version

# CUDA 11.x+ (GPU训练)
nvidia-smi
```

**推荐配置**:
- GPU: NVIDIA RTX 3060 (12GB) 或更高
- 内存: 16GB+
- 存储: 50GB+ SSD

### 2. 安装依赖

```bash
cd Farmland_Crack_Detection
pip install -r requirements.txt
```

**核心依赖**:
```txt
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
albumentations>=1.3.0
matplotlib>=3.7.0
```

### 3. 数据准备

项目已包含完整的极限增强数据集，位于 `data/` 目录：

```
data/
├── data.yaml              # 数据集配置文件
├── train/
│   ├── images/            # 训练图片 (~1040张)
│   └── labels/            # YOLO格式标签
├── val/
│   ├── images/            # 验证图片 (~130张)
│   └── labels/
└── test/
    ├── images/            # 测试图片 (~131张)
    └── labels/
```

**类别定义**:

| 类别ID | 类别名称 | 英文标识 | 描述 |
|--------|----------|----------|------|
| 0 | 细微裂纹 | tiny_crack | 裂缝宽度 < 5mm |
| 1 | 网状裂隙 | mesh_crack | 网状分布的裂隙 |
| 2 | 深大裂缝 | deep_crack | 裂缝宽度 > 20mm |

### 4. 开始训练

#### 方式一：直接运行训练脚本（推荐）

```bash
cd Farmland_Crack_Detection/scripts
python train_extreme.py
```

训练脚本会自动：
- ✅ 配置数据集路径
- ✅ 加载 YOLOv10m 预训练权重
- ✅ 执行200轮训练（带早停机制）
- ✅ 应用极限数据增强策略
- ✅ 训练完成后自动执行 TTA 评估
- ✅ 生成完整可视化报告

#### 方式二：手动配置训练参数

```python
from ultralytics import YOLO

model = YOLO('yolov10m.pt')

results = model.train(
    data='data/data.yaml',
    epochs=200,
    imgsz=1024,
    batch=4,
    device=0,
    lr0=0.001,
    optimizer='SGD',
    mosaic=0.9,
    mixup=0.15,
    copy_paste=0.3,
    patience=40,
    project='runs',
    name='extreme_m'
)
```

### 5. 查看结果

训练完成后，输出目录结构：

```
runs/extreme_m/
├── weights/
│   ├── best.pt              # 最佳模型 (按mAP@0.5选择)
│   └── last.pt              # 最后一个epoch的模型
├── results.csv              # 详细训练指标记录
├── results.png              # 训练曲线和指标
├── confusion_matrix.png     # 混淆矩阵
├── confusion_matrix_normalized.png  # 归一化混淆矩阵
├── BoxPR_curve.png          # Precision-Recall曲线
├── BoxP_curve.png           # Precision曲线
├── BoxR_curve.png           # Recall曲线
├── BoxF1_curve.png          # F1-Score曲线
├── labels.jpg               # 数据集标注可视化
├── val_batch*_labels.jpg    # 验证集真实标签
└── val_batch*_pred.jpg      # 验证集预测结果

runs/extreme_m_tta_test/   # TTA测试集评估结果
runs/extreme_m_tta_val/    # TTA验证集评估结果
```

---

## ⚙️ 训练配置详解

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **模型** | YOLOv10m | Medium版本，精度与速度平衡 |
| **图像尺寸** | 1024×1024 | 高分辨率提升小目标检测 |
| **训练轮次** | 200 | 最大epoch数（实际可能提前停止） |
| **批大小** | 4 | 根据显存调整（12GB显存推荐） |
| **初始学习率** | 0.001 | SGD优化器 |
| **学习率策略** | 余弦退火 + Warmup | 前5轮warmup |
| **早停耐心值** | 40 | 连续40轮无改善则停止 |

### 极限数据增强策略

| 增强方法 | 概率/参数 | 效果 |
|----------|-----------|------|
| **Mosaic** | 0.9 | 4图拼接，增加目标多样性 |
| **MixUp** | 0.15 | 图像混合，提升泛化能力 |
| **Copy-Paste** | 0.3 | 复制粘贴目标，增加小目标样本 |
| **水平翻转** | 0.5 | 镜像对称 |
| **垂直翻转** | 0.5 | 上下翻转 |
| **旋转角度** | ±30° | 大幅旋转增强 |
| **缩放范围** | 0.6 | 多尺度适应 |
| **剪切变换** | 10° | 形变增强 |
| **透视变换** | 0.001 | 3D视角模拟 |
| **H-S-V色彩** | 0.03/0.8/0.5 | 强色彩扰动 |
| **随机擦除** | 0.3 | 遮挡模拟 |
| **关闭Mosaic** | 第40轮后 | 后期稳定训练 |

### 优化器配置

```yaml
optimizer: SGD
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5
warmup_momentum: 0.8
warmup_bias_lr: 0.1
cos_lr: true
```

---

## 📊 评估指标说明

### 核心指标

| 指标 | 英文全称 | 中文 | 说明 |
|------|----------|------|------|
| **Precision** | Precision | 精确率/查准率 | 预测为正样本中真正本的比例 |
| **Recall** | Recall | 召回率/查全率 | 实际正样本中被正确预测的比例 |
| **mAP50** | mean Average Precision @ IoU=0.5 | 平均精度均值 | IoU阈值0.5时的平均精度 |
| **mAP50-95** | mAP@[0.5:0.95] | 多尺度平均精度 | IoU从0.5到0.95的平均精度 |

### 可视化报告文件

| 文件名 | 内容描述 |
|--------|----------|
| `results.png` | 训练损失曲线、mAP/Precision/Recall/F1曲线、学习率变化 |
| `confusion_matrix.png` | 测试集混淆矩阵（原始计数） |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵（比例显示） |
| `BoxPR_curve.png` | Precision-Recall曲线（含各类别AP@50值） |
| `BoxP_curve.png` | Precision随置信度阈值变化曲线 |
| `BoxR_curve.png` | Recall随置信度阈值变化曲线 |
| `BoxF1_curve.png` | F1-Score随置信度阈值变化曲线 |
| `labels.jpg` | 数据集样本标注可视化（6张示例） |

---

## 🔧 推理与部署

### 使用最佳模型推理

```python
from ultralytics import YOLO

model = YOLO('runs/extreme_m/weights/best.pt')

results = model.predict(
    source='your_image.jpg',
    imgsz=1024,
    conf=0.25,        # 置信度阈值
    iou=0.5,          # NMS IoU阈值
    save=True         # 保存结果
)

for r in results:
    for box in r.boxes:
        print(f"类别: {box.cls[0]}, 置信度: {box.conf[0]:.4f}")
```

### 批量推理

```python
model.predict(
    source='test_images/',
    imgsz=1024,
    conf=0.25,
    save=True,
    project='results/',
    name='predictions'
)
```

---

## 📁 项目结构

```
Farmland_Crack_Detection/
├── .gitignore                  # Git忽略规则
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖列表
├── convert_weights.py          # 权重转换工具（可选）
│
├── data/                       # 数据集目录
│   ├── data.yaml               # 数据集配置文件
│   ├── train/                  # 训练集 (~1040张)
│   │   ├── images/
│   │   └── labels/
│   ├── val/                    # 验证集 (~130张)
│   │   ├── images/
│   │   └── labels/
│   └── test/                   # 测试集 (~131张)
│       ├── images/
│       └── labels/
│
├── scripts/                    # 脚本目录
│   └── train_extreme.py        # 主训练脚本（含TTA评估）
│
└── runs/                       # 运行结果（自动生成）
    └── farmland/
        ├── extreme_m/          # 主训练结果
        │   ├── weights/best.pt
        │   ├── results.csv
        │   ├── confusion_matrix.png
        │   └── ...
        ├── extreme_m_tta_test/ # TTA测试集评估
        └── extreme_m_tta_val/  # TTA验证集评估
```

---

## ⚠️ 常见问题

### Q1: 显存不足 (OOM)

**解决方案**：
1. 减小批大小：
   ```python
   batch=2  # 或 1
   ```
2. 减小图像尺寸：
   ```python
   imgsz=800  # 或 640
   ```

**显存参考**：
| 配置 | 预估显存 | 推荐GPU |
|------|----------|---------|
| batch=4, imgsz=1024 | ~10GB | RTX 3060 (12GB) |
| batch=4, imgsz=800 | ~6GB | GTX 1660 (6GB) |
| batch=2, imgsz=640 | ~3GB | GTX 1650 (4GB) |

### Q2: 训练速度慢

**检查项**：
```bash
# 确认GPU使用
nvidia-smi

# 验证CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 预训练权重下载失败

**手动下载**：
```bash
# 从GitHub下载YOLOv10m预训练权重
wget https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10m.pt
mv yolov10m.pt Farmland_Crack_Detection/
```

### Q4: Linux下中文显示乱码

安装中文字体：
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# 清除matplotlib缓存
rm -rf ~/.cache/matplotlib
```

### Q5: 如何使用自己的数据集？

1. 准备YOLO格式数据集（images + labels）
2. 修改 `data/data.yaml` 中的路径和类别
3. 运行训练脚本

---

## 🎯 使用流程总结

```mermaid
graph LR
    A[环境准备<br>pip install -r requirements.txt] --> B[数据检查<br>data/ 目录完整性]
    B --> C[运行训练<br>python scripts/train_extreme.py]
    C --> D[自动TTA评估<br>test + val集合]
    D --> E[查看结果<br>runs/extreme_m/]
    E --> F[模型部署<br>best.pt 推理]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    E fill:#ffebee
    F fill:#fce4ec
```

---

## 📝 注意事项

1. **首次运行**: 需要下载YOLOv10m预训练权重（约100MB）
2. **显存需求**: 推荐12GB+显存运行默认配置
3. **训练时间**: 约3-6小时（取决于GPU和数据量）
4. **早停机制**: 可能提前于200轮停止（patience=40）
5. **数据增强**: 第40轮后自动关闭Mosaic/MixUp以稳定训练
6. **TTA警告**: 如提示"Model does not support augment=True"，属正常现象，不影响评估

---

## 📞 技术支持

如遇问题，请检查：
1. 日志文件: `runs/extreme_m/results.csv`
2. 终端错误信息
3. GPU状态: `nvidia-smi`
4. 数据集完整性: `data/train/images/` 数量

---

**最后更新**: 2026-05-16  
**适用环境**: Ubuntu 20.04+, Windows 10/11, AutoDL  
**Python版本**: 3.8 - 3.11  
**PyTorch版本**: >= 2.0.0  
**ULtralytics版本**: >= 8.0.0