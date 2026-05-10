# Code Wiki — 基于改进YOLOv10的农田干裂智能识别系统

> 项目名称：基于改进YOLOv10的农田干裂智能识别原型系统研究与实现  
> 作者：高一峰 | 指导教师：田颖  
> 技术栈：Python 3.11 / PyTorch / PyQt5 / OpenCV / Albumentations

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 [模型模块 (models)](#41-模型模块-models)
   - 4.2 [训练模块 (train.py)](#42-训练模块-trainpy)
   - 4.3 [推理模块 (inference.py)](#43-推理模块-inferencepy)
   - 4.4 [评估模块 (evaluate.py)](#44-评估模块-evaluatepy)
   - 4.5 [数据集模块 (utils/roboflow_dataset.py)](#45-数据集模块-utilsroboflow_datasetpy)
   - 4.6 [工具模块 (utils)](#46-工具模块-utils)
   - 4.7 [GUI模块 (gui)](#47-gui模块-gui)
   - 4.8 [辅助工具集 (tools)](#48-辅助工具集-tools)
5. [关键类与函数索引](#5-关键类与函数索引)
6. [依赖关系图](#6-依赖关系图)
7. [配置文件说明](#7-配置文件说明)
8. [项目运行方式](#8-项目运行方式)
9. [数据集说明](#9-数据集说明)

---

## 1. 项目概述

本项目是一个基于改进 YOLOv10 的农田干裂程度智能识别系统，核心改进点包括：

- **CSPDarknet 骨干网络**：采用 YOLOv10s 的 CSPDarknet 结构，支持深度/宽度缩放因子
- **PANet 特征融合**：双向特征金字塔网络，实现多尺度特征高效融合
- **CBAM 注意力机制**：在 Neck 输出后引入通道-空间注意力模块，增强裂缝特征表达
- **DecoupledHead 解耦头**：分类与回归分支解耦，回归分支采用 DFL (Distribution Focal Loss) 编码
- **Anchor-Free 检测范式**：无需预设锚框，通过网格中心点直接回归边界框
- **Sigmoid Focal Loss**：解决正负样本极度不平衡问题
- **CIoU Loss**：边界框回归采用 Complete IoU 损失

检测类别为三级干裂程度：

| 类别ID | 类别名称 | 说明 | 颜色标识 |
|--------|----------|------|----------|
| 0 | 细微裂纹 | 裂缝宽度 < 5mm | 绿色 |
| 1 | 网状裂隙 | 裂缝宽度 5-20mm | 黄色 |
| 2 | 深大裂缝 | 裂缝宽度 > 20mm | 红色 |

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    农田干裂识别系统                        │
├─────────────┬───────────────────┬───────────────────────┤
│   数据层     │     模型层         │      应用层            │
├─────────────┼───────────────────┼───────────────────────┤
│ Roboflow    │  Backbone         │  GUI (PyQt5)          │
│ Dataset     │  (CSPDarknet)     │  MainWindow           │
│ Adapter     │       ↓           │  InferenceThread      │
│             │  Neck (PANet)     │                       │
│ Albument-   │       ↓           │  CLI 推理接口          │
│ ations增强   │  CBAM Attention   │  CrackDetector        │
│             │       ↓           │                       │
│ YOLO格式    │  DecoupledHead    │  辅助工具集            │
│ 标签解析     │  (Cls + Reg)      │  (标注/下载/过滤等)    │
├─────────────┼───────────────────┼───────────────────────┤
│   训练管线   │     评估管线       │      部署管线          │
├─────────────┼───────────────────┼───────────────────────┤
│ Trainer     │  evaluate_map()   │  TorchScript          │
│ - FocalLoss │  calculate_map()  │  ONNX (预留)          │
│ - CIoULoss  │  Precision/Recall │                       │
│ - DFLLoss   │  mAP@0.5         │                       │
│ - AMP       │                   │                       │
│ - GradAccum │                   │                       │
└─────────────┴───────────────────┴───────────────────────┘
```

### 模型推理数据流

```
输入图像 (H×W×3)
    │
    ▼
ImagePreprocessor.preprocess_pipeline()
    ├── color_correction()    → CLAHE 色彩校正
    ├── denoise()             → 快速非局部均值去噪
    └── resize_with_padding() → 等比缩放 + 填充
    │
    ▼
YOLOv10Crack.forward()
    ├── Backbone: CSPDarknet → [P2, P3, P4]  (stride 8/16/32)
    ├── Neck:     PANet      → [N3, N4, N5]  (256通道)
    ├── Attention: CBAM × 3  → 增强特征
    └── Head:     DecoupledHead × 3
         ├── cls_pred: [B, NC, H, W]     分类预测
         └── reg_pred: [B, 4*reg_max, H, W]  回归预测
    │
    ▼
_decode_outputs()
    ├── DFL 解码: softmax → 期望值
    ├── 网格坐标 → 绝对坐标
    └── 拼接 [boxes, cls_scores]
    │
    ▼
CrackDetector.postprocess()
    ├── 置信度过滤
    ├── 逐类 NMS
    └── 坐标还原 (去padding/缩放)
    │
    ▼
ResultVisualizer.draw_detections()
    └── 绘制检测框 + 标签
```

---

## 3. 目录结构

```
YOLOv10/
├── Farmland_Crack_Detection/          # 核心项目目录
│   ├── models/                        # 模型定义
│   │   ├── __init__.py                # 导出 YOLOv10Crack, CBAM 等
│   │   ├── yolov10_crack.py           # 主模型类
│   │   ├── attention.py               # CBAM 注意力模块
│   │   ├── backbone/
│   │   │   ├── __init__.py            # 导出 CSPDarknet, ConvBlock, C2f, SPPF
│   │   │   ├── csp_darknet.py         # CSPDarknet 骨干网络
│   │   │   └── csp_darknet_fixed.py   # 骨干网络修正版
│   │   ├── neck/
│   │   │   ├── __init__.py            # 导出 PANet
│   │   │   ├── panet.py               # PANet 特征融合
│   │   │   └── panet_fixed.py         # PANet 修正版
│   │   └── head/
│   │       ├── __init__.py            # 导出 DecoupledHead
│   │       └── decoupled_head.py      # 解耦检测头
│   ├── utils/                         # 工具函数
│   │   ├── __init__.py                # 统一导出
│   │   ├── roboflow_dataset.py        # 数据集适配器
│   │   ├── data_processing.py         # 图像预处理
│   │   ├── visualization.py           # 结果可视化
│   │   ├── metrics.py                 # 评估指标
│   │   └── logger.py                  # 日志工具
│   ├── gui/                           # 图形界面
│   │   ├── __init__.py                # 导出 MainWindow
│   │   ├── main_window.py             # PyQt5 主窗口
│   │   └── widgets/                   # 自定义控件
│   ├── configs/                       # 配置文件
│   │   ├── model/
│   │   │   └── yolov10_crack.yaml     # 模型结构配置
│   │   ├── train.yaml                 # 基础训练配置
│   │   ├── train_farmland_finetune.yaml  # 农田微调配置
│   │   ├── train_road_crack.yaml      # 道路裂缝预训练配置
│   │   ├── train_personal.yaml        # 个人配置模板
│   │   └── inference.yaml             # 推理配置
│   ├── data/                          # 数据集目录
│   │   ├── dataset.yaml               # 数据集配置
│   │   ├── train/                     # 训练集
│   │   ├── val/                       # 验证集
│   │   ├── test/                      # 测试集
│   │   └── new_downloads/             # 新下载待标注数据
│   ├── tests/                         # 单元测试
│   │   ├── test_data.py               # 数据模块测试
│   │   ├── test_gui.py                # GUI模块测试
│   │   └── test_model.py              # 模型模块测试
│   ├── tools/                         # 内部调试工具
│   │   ├── analyze_dataset.py         # 数据集分析
│   │   ├── batch_test.py              # 批量测试
│   │   ├── check_*.py                 # 各类检查脚本
│   │   ├── debug_anchor.py            # Anchor调试
│   │   ├── rebuild_dataset*.py        # 数据集重建
│   │   ├── split_dataset.py           # 数据集划分
│   │   ├── test_*.py                  # 测试脚本
│   │   └── ...
│   ├── scripts/
│   │   ├── setup_env.bat              # Windows环境配置
│   │   └── setup_env.sh               # Linux环境配置
│   ├── docs/                          # 文档
│   ├── train.py                       # 训练入口
│   ├── train_single.py                # 单图训练调试
│   ├── inference.py                   # 推理入口
│   ├── evaluate.py                    # 评估入口
│   ├── convert_weights.py             # 权重转换
│   ├── analyze_weights.py             # 权重分析
│   ├── requirements.txt               # Python依赖
│   └── .gitignore                     # Git忽略规则
│
├── tools/                             # 外部辅助工具集
│   ├── assisted_annotation.py         # 半自动标注工具 (tkinter)
│   ├── auto_label.py                  # 自动伪标签生成
│   ├── check_dataset.py               # 数据集完整性检查
│   ├── distribute_new_data.py         # 新数据分配工具
│   ├── download_farmland_images.py    # 农田图像下载器
│   ├── evaluate_model.py              # 模型评估工具
│   ├── filter_china_drone.py          # 无人机数据过滤
│   ├── review_labels.py               # 标注审查工具
│   ├── split_farmland_dataset.py      # 数据集划分工具
│   └── visualize_dataset.py           # 数据集可视化 (tkinter)
│
├── filtered_china_drone/              # 过滤后的无人机数据集
│   └── dataset.yaml
│
├── China_Drone/                       # 原始无人机数据集
└── review_data/                       # 审查数据
```

---

## 4. 核心模块详解

### 4.1 模型模块 (models)

#### 4.1.1 YOLOv10Crack — 主模型类

**文件**: [models/yolov10_crack.py](Farmland_Crack_Detection/models/yolov10_crack.py)

```python
class YOLOv10Crack(nn.Module)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_classes` | int | 3 | 检测类别数 |
| `depth_multiple` | float | 0.33 | 深度缩放因子 (YOLOv10s) |
| `width_multiple` | float | 0.50 | 宽度缩放因子 (YOLOv10s) |
| `use_attention` | bool | True | 是否使用CBAM注意力 |
| `reg_max` | int | 16 | DFL回归最大值 |

**模型组成**:

| 组件 | 类 | 输出通道 | 说明 |
|------|-----|----------|------|
| backbone | CSPDarknet | [64, 128, 256] (×0.50) | 4阶段特征提取 + SPPF |
| neck | PANet | [256, 256, 256] | 双向特征金字塔 |
| attention | CBAM × 3 | 256 | 通道+空间注意力 |
| heads | DecoupledHead × 3 | cls+reg | 解耦检测头 |

**前向传播逻辑**:

- **训练模式** (`self.training=True`): 返回 `List[Tuple[cls_pred, reg_pred]]`，每个元素对应一个 stride (8/16/32)
- **推理模式** (`self.training=False`): 返回 `Tensor [B, N, 4+NC]`，经 DFL 解码后的拼接预测

**关键方法**:

| 方法 | 说明 |
|------|------|
| `forward(x, return_raw=False)` | 前向传播，return_raw 强制返回原始输出 |
| `_decode_outputs(outputs, input_shape)` | DFL 解码 + 网格坐标转换 → 绝对坐标 |

---

#### 4.1.2 CSPDarknet — 骨干网络

**文件**: [models/backbone/csp_darknet.py](Farmland_Crack_Detection/models/backbone/csp_darknet.py)

```python
class CSPDarknet(nn.Module)
```

5 阶段下采样结构，输出 3 个尺度特征图：

| 阶段 | 输出步长 | 输出通道 (×0.50) | 组成 |
|------|----------|-------------------|------|
| stem | /2 | 32 | ConvBlock(3→32, k=3, s=2) |
| stage1 | /4 | 64 | ConvBlock + C2f(depth=1) |
| stage2 | /8 | 128 | ConvBlock + C2f(depth=2) |
| stage3 | /16 | 256 | ConvBlock + C2f(depth=3) |
| stage4 | /32 | 512 | ConvBlock + C2f(depth=1) + SPPF |

**输出**: `[stage2_out, stage3_out, stage4_out]` → 对应 stride 8/16/32

**子模块**:

| 类 | 说明 |
|-----|------|
| `ConvBlock` | 标准卷积块: Conv2d → BN → SiLU |
| `C2f` | CSP瓶颈层: 通道分割 + 多分支卷积 + 拼接融合 |
| `SPPF` | 空间金字塔池化: 3次MaxPool2d串联 → 4路拼接 |

---

#### 4.1.3 PANet — 特征融合网络

**文件**: [models/neck/panet.py](Farmland_Crack_Detection/models/neck/panet.py)

```python
class PANet(nn.Module)
```

双向特征金字塔，融合骨干网络的三尺度特征：

```
P5 ──Upsample──→ Cat(P4) ──C2f──→ N4 ──Upsample──→ Cat(P3) ──C2f──→ N3
                                                                    │
N3 ──ConvBlock(s=2)──→ Cat(N4) ──C2f──→ N4' ──ConvBlock(s=2)──→ Cat(P5) ──C2f──→ N5'
```

**输出**: `[N3, N4', N5']`，均为 256 通道

---

#### 4.1.4 CBAM — 注意力模块

**文件**: [models/attention.py](Farmland_Crack_Detection/models/attention.py)

```python
class CBAM(nn.Module)
```

通道-空间双重注意力机制，串联应用于 Neck 输出的每个尺度：

| 子模块 | 说明 |
|--------|------|
| `ChannelAttention` | AvgPool + MaxPool → 共享FC → Sigmoid 门控 |
| `SpatialAttention` | 通道均值+最大值拼接 → Conv7×7 → Sigmoid 门控 |

**参数**: `reduction=16` (通道压缩比), `kernel_size=7` (空间卷积核)

---

#### 4.1.5 DecoupledHead — 解耦检测头

**文件**: [models/head/decoupled_head.py](Farmland_Crack_Detection/models/head/decoupled_head.py)

```python
class DecoupledHead(nn.Module)
```

分类与回归分支完全解耦：

| 分支 | 结构 | 输出 |
|------|------|------|
| cls_conv | DepthwiseConv → PointwiseConv → BN → SiLU | - |
| cls_pred | Conv2d(in_ch, num_classes, 1) | `[B, NC, H, W]` |
| reg_conv | DepthwiseConv → PointwiseConv → BN → SiLU | - |
| reg_pred | Conv2d(in_ch, 4*reg_max, 1) | `[B, 4*reg_max, H, W]` |

---

### 4.2 训练模块 (train.py)

**文件**: [train.py](Farmland_Crack_Detection/train.py)

#### Trainer 类

```python
class Trainer:
    def __init__(self, config_path: str)
```

**初始化流程**:

1. 加载 YAML 配置
2. `_init_model()` — 创建模型 + 加载预训练权重（兼容形状匹配）
3. `_init_data()` — 创建训练/验证 DataLoader + 加权采样
4. `_init_training()` — 配置优化器/调度器/AMP/梯度累积

**损失函数**:

| 损失 | 方法 | 说明 |
|------|------|------|
| 分类损失 | `_sigmoid_focal_loss()` | Focal Loss (α=0.25, γ=1.5) + 类别权重 |
| 回归损失 | `_bbox_ciou_loss()` | Complete IoU Loss |
| DFL损失 | 内联于 `_compute_loss()` | Distribution Focal Loss |

**训练特性**:

| 特性 | 配置键 | 说明 |
|------|--------|------|
| 混合精度 | `training.amp.enabled` | GradScaler + autocast |
| 梯度累积 | `training.gradient_accumulation` | 模拟更大 batch_size |
| 学习率预热 | `training.scheduler.warmup_epochs` | 线性预热 |
| 加权采样 | `training.sampling.enabled` | WeightedRandomSampler |
| 梯度裁剪 | `training.grad_clip_norm` | 防止梯度爆炸 |
| 标签平滑 | `training.loss.label_smoothing` | 正则化 |

**保存策略**:

- `best_loss.pt` — 验证损失最低
- `best.pt` — mAP@0.5 最高
- `epoch_N.pt` — 周期性保存
- `last.pt` — 最终模型

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/train.yaml` | 配置文件路径 |
| `--validate` | False | 仅运行验证 |
| `--weights` | `outputs/exp_anchor_loss/weights/best.pt` | 验证用权重路径 |

---

### 4.3 推理模块 (inference.py)

**文件**: [inference.py](Farmland_Crack_Detection/inference.py)

#### CrackDetector 类

```python
class CrackDetector:
    def __init__(self, model_path, data_yaml=None, device='cuda',
                 conf_threshold=0.5, iou_threshold=0.45, input_size=512)
```

**推理流程**:

1. `detect(image)` → 图像预处理 → 模型推理 → 后处理
2. `detect_and_visualize(image)` → 检测 + 可视化

**后处理流程** (`postprocess()`):

1. 置信度过滤
2. 逐类别 NMS
3. 坐标还原（去 padding / 缩放回原图尺寸）

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | 必填 | 输入图像路径 |
| `--model` | `weights/best_model.pt` | 模型权重路径 |
| `--data` | `data/dataset.yaml` | 数据集配置 |
| `--output` | `output.jpg` | 输出图像路径 |
| `--conf` | 0.5 | 置信度阈值 |
| `--iou` | 0.45 | NMS IoU阈值 |
| `--img-size` | 512 | 输入图像尺寸 |

---

### 4.4 评估模块 (evaluate.py)

**文件**: [evaluate.py](Farmland_Crack_Detection/evaluate.py)

#### 核心函数

| 函数 | 说明 |
|------|------|
| `evaluate_map(model, dataloader, device, ...)` | 计算 mAP@IoU、Precision、Recall、Per-class AP |
| `decode_predictions(predictions, conf_threshold, iou_threshold)` | 解码模型输出为检测列表 |
| `build_model(weights_path, device, num_classes)` | 构建模型并加载权重 |
| `nms(boxes, scores, iou_threshold)` | 非极大值抑制 |

**评估指标**:

- **mAP@0.5**: 多类别平均精度均值
- **Precision**: 精确率
- **Recall**: 召回率
- **Per-class AP**: 每类平均精度

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `outputs/exp_anchor_loss/weights/best.pt` | 模型权重 |
| `--data` | `data/dataset.yaml` | 数据集配置 |
| `--conf` | 0.5 | 置信度阈值 |
| `--map-conf` | 0.05 | mAP计算用置信度 |
| `--iou` | 0.5 | IoU阈值 |
| `--img-size` | 训练时尺寸 | 评估图像尺寸 |
| `--batch-size` | 1 | 评估批次大小 |

---

### 4.5 数据集模块 (utils/roboflow_dataset.py)

**文件**: [utils/roboflow_dataset.py](Farmland_Crack_Detection/utils/roboflow_dataset.py)

#### RoboflowFarmlandDataset 类

```python
class RoboflowFarmlandDataset(Dataset):
    def __init__(self, data_yaml_path, split="train",
                 image_size=(512, 512), augment=True, cache_images=False)
```

**核心功能**:

| 方法 | 说明 |
|------|------|
| `__getitem__(idx)` | 返回 `{'image', 'labels', 'image_path', 'original_size', 'num_objects'}` |
| `get_class_weights()` | 计算逆频率类别权重 |
| `get_image_sampling_weights(...)` | 计算每张图像的采样权重（支持类别增强） |
| `visualize_sample(idx, save_path)` | 可视化单个样本 |
| `_load_image(image_path)` | PIL优先 → OpenCV备选加载 |
| `_load_labels(label_path, w, h)` | YOLO格式 → 像素坐标 [x1,y1,x2,y2,cls] |

**数据增强** (Albumentations):

训练集增强策略（偏保守，保护细裂缝特征）：

| 增强方法 | 参数 | 概率 |
|----------|------|------|
| ShiftScaleRotate | shift=0.03, scale=0.05, rotate=4° | 0.20 |
| RandomBrightnessContrast | brightness=0.08, contrast=0.08 | 0.15 |
| CLAHE | clip_limit=2.0 | 0.10 |

验证/测试集：仅 Resize + Normalize

**标签格式**: YOLO格式 `class_id x_center y_center width height` (归一化坐标)

**辅助函数**:

| 函数 | 说明 |
|------|------|
| `create_farmland_dataloaders(...)` | 一次性创建 train/val/test DataLoader |

---

### 4.6 工具模块 (utils)

#### ImagePreprocessor — 图像预处理

**文件**: [utils/data_processing.py](Farmland_Crack_Detection/utils/data_processing.py)

| 方法 | 说明 |
|------|------|
| `resize_image(image)` | 等比缩放 + 左上角对齐填充 |
| `resize_with_padding(image)` | 等比缩放 + 居中填充，返回 transform_info |
| `normalize_image(image)` | 归一化到 [0, 1] |
| `convert_color_space(image, src, dst)` | BGR ↔ RGB 转换 |
| `color_correction(image)` | CLAHE 色彩校正 (LAB空间) |
| `denoise(image, method)` | 去噪: fast / bilateral / gaussian |
| `preprocess_pipeline(image)` | 完整管线: 校正 → 去噪 → 缩放填充 |

#### ResultVisualizer — 结果可视化

**文件**: [utils/visualization.py](Farmland_Crack_Detection/utils/visualization.py)

| 方法 | 说明 |
|------|------|
| `draw_detections(image, detections)` | 绘制检测框 + 类别标签 + 置信度 |
| `draw_statistics_panel(image, detections)` | 拼接统计面板 |
| `create_detection_report(detections, output_path)` | 生成饼图+柱状图报告 |
| `export_results(detections, image_path, format)` | 导出 JSON 格式结果 |

#### metrics — 评估指标

**文件**: [utils/metrics.py](Farmland_Crack_Detection/utils/metrics.py)

| 函数 | 说明 |
|------|------|
| `calculate_iou(box1, box2)` | 计算两个边界框的 IoU |
| `calculate_ap(recalls, precisions)` | 计算 Average Precision (11点插值) |
| `calculate_map(predictions, targets, iou_thresholds, num_classes)` | 计算 mAP 和 Per-class AP |
| `calculate_precision_recall(predictions, targets, iou_threshold)` | 计算 Precision / Recall / TP / FP / FN |

#### logger — 日志工具

**文件**: [utils/logger.py](Farmland_Crack_Detection/utils/logger.py)

| 函数/类 | 说明 |
|---------|------|
| `setup_logger(name, log_dir, level)` | 配置 Logger (控制台 + 文件) |
| `AverageMeter` | 滑动平均值计算器 |

---

### 4.7 GUI模块 (gui)

**文件**: [gui/main_window.py](Farmland_Crack_Detection/gui/main_window.py)

#### MainWindow 类

```python
class MainWindow(QMainWindow)
```

基于 PyQt5 的桌面检测界面，功能包括：

| 功能 | 说明 |
|------|------|
| 模型加载 | 自动扫描 outputs/ 目录下的 best.pt |
| 单图检测 | 加载单张图像进行推理 |
| 批量检测 | 加载文件夹，支持上下翻页 |
| 参数调节 | 置信度阈值 / IoU 阈值 |
| 结果导出 | 导出带检测框的图像 |

#### InferenceThread 类

```python
class InferenceThread(QThread)
```

异步推理线程，避免 GUI 卡顿：

- 信号: `finished(list, np.ndarray)` / `progress(int)`
- 内置 NMS 实现

---

### 4.8 辅助工具集 (tools)

位于 `YOLOv10/tools/` 目录，提供数据采集、标注、质量检查等辅助功能：

| 脚本 | 功能 | 界面 |
|------|------|------|
| `assisted_annotation.py` | 半自动标注：模型预测 + 人工修正 | tkinter |
| `auto_label.py` | 自动伪标签生成 | CLI |
| `check_dataset.py` | 数据集完整性检查 | CLI |
| `distribute_new_data.py` | 新数据按比例分配到 train/val/test | CLI |
| `download_farmland_images.py` | 从 Pexels/Pixabay 下载农田图像 | CLI |
| `evaluate_model.py` | 模型评估（P/R/F1） | CLI |
| `filter_china_drone.py` | 过滤无人机数据集，XML→YOLO格式 | CLI |
| `review_labels.py` | 标注审查，生成预览图 | CLI |
| `split_farmland_dataset.py` | 单一训练集划分为 train/val/test | CLI |
| `visualize_dataset.py` | 数据集可视化浏览 | tkinter |

---

## 5. 关键类与函数索引

### 模型类

| 类名 | 文件 | 说明 |
|------|------|------|
| `YOLOv10Crack` | models/yolov10_crack.py | 主检测模型 |
| `CSPDarknet` | models/backbone/csp_darknet.py | 骨干网络 |
| `PANet` | models/neck/panet.py | 特征融合网络 |
| `DecoupledHead` | models/head/decoupled_head.py | 解耦检测头 |
| `CBAM` | models/attention.py | 通道-空间注意力 |
| `ChannelAttention` | models/attention.py | 通道注意力子模块 |
| `SpatialAttention` | models/attention.py | 空间注意力子模块 |
| `ConvBlock` | models/backbone/csp_darknet.py | 标准卷积块 |
| `C2f` | models/backbone/csp_darknet.py | CSP瓶颈层 |
| `SPPF` | models/backbone/csp_darknet.py | 空间金字塔池化 |

### 训练/推理/评估类

| 类名 | 文件 | 说明 |
|------|------|------|
| `Trainer` | train.py | 训练管理器 |
| `CrackDetector` | inference.py | 推理检测器 |

### 数据类

| 类名 | 文件 | 说明 |
|------|------|------|
| `RoboflowFarmlandDataset` | utils/roboflow_dataset.py | 数据集适配器 |
| `ImagePreprocessor` | utils/data_processing.py | 图像预处理器 |
| `ResultVisualizer` | utils/visualization.py | 结果可视化器 |
| `AverageMeter` | utils/logger.py | 滑动平均计算器 |

### GUI类

| 类名 | 文件 | 说明 |
|------|------|------|
| `MainWindow` | gui/main_window.py | PyQt5主窗口 |
| `InferenceThread` | gui/main_window.py | 异步推理线程 |

### 关键函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `evaluate_map()` | evaluate.py | mAP评估主函数 |
| `calculate_map()` | utils/metrics.py | mAP计算 |
| `calculate_precision_recall()` | utils/metrics.py | P/R计算 |
| `calculate_ap()` | utils/metrics.py | AP计算 |
| `setup_logger()` | utils/logger.py | 日志配置 |
| `create_farmland_dataloaders()` | utils/roboflow_dataset.py | DataLoader工厂 |
| `collate_fn()` | train.py / evaluate.py | 自定义批次合并 |

---

## 6. 依赖关系图

### 模块依赖

```
train.py
├── models.yolov10_crack.YOLOv10Crack
│   ├── models.backbone.csp_darknet.CSPDarknet
│   │   ├── ConvBlock
│   │   ├── C2f
│   │   └── SPPF
│   ├── models.neck.panet.PANet
│   │   ├── ConvBlock (from backbone)
│   │   └── C2f (from backbone)
│   ├── models.attention.CBAM
│   │   ├── ChannelAttention
│   │   └── SpatialAttention
│   └── models.head.decoupled_head.DecoupledHead
├── evaluate.evaluate_map
│   ├── utils.metrics.calculate_map
│   └── utils.metrics.calculate_precision_recall
├── utils.roboflow_dataset.RoboflowFarmlandDataset
└── utils.logger.setup_logger

inference.py
├── models.yolov10_crack.YOLOv10Crack
├── utils.data_processing.ImagePreprocessor
└── utils.visualization.ResultVisualizer

evaluate.py
├── models.yolov10_crack.YOLOv10Crack
├── utils.metrics.calculate_map / calculate_precision_recall
└── utils.roboflow_dataset.RoboflowFarmlandDataset

gui/main_window.py
└── models.yolov10_crack.YOLOv10Crack
```

### Python包依赖

```
核心依赖:
├── torch >= 2.0.0          # 深度学习框架
├── torchvision >= 0.15.0   # 视觉工具包
├── ultralytics >= 8.0.0    # YOLO官方库(权重转换用)
├── opencv-python >= 4.8.0  # 图像处理
├── Pillow >= 10.0.0        # 图像读取
├── numpy >= 1.24.0         # 数值计算
├── albumentations >= 1.3.0 # 数据增强
└── PyYAML >= 6.0           # 配置文件解析

评估与可视化:
├── scipy >= 1.11.0         # 科学计算
├── pandas >= 2.0.0         # 数据分析
├── matplotlib >= 3.7.0     # 绘图
├── seaborn >= 0.12.0       # 统计可视化
└── pycocotools >= 2.0.6    # COCO评估工具

GUI:
├── PyQt5 >= 5.15.0         # 桌面界面
├── PyQt5-Qt5 >= 5.15.0
└── PyQt5-sip >= 12.12.0

训练辅助:
├── tensorboard >= 2.14.0   # 训练可视化
├── tqdm >= 4.66.0          # 进度条
└── requests >= 2.31.0      # HTTP请求(图像下载)
```

---

## 7. 配置文件说明

### 训练配置层级

项目提供多级训练配置，支持渐进式训练策略：

| 配置文件 | 用途 | 关键差异 |
|----------|------|----------|
| `configs/train_road_crack.yaml` | 道路裂缝预训练 | 640×640, batch=8, lr=0.0008, 150 epochs |
| `configs/train_farmland_finetune.yaml` | 农田数据微调 | 800×800, batch=2, lr=0.00008, 60 epochs, 加权采样 |
| `configs/train.yaml` | 基础训练 | 512×512, batch=4, lr=0.0005, 100 epochs |

### 配置结构

```yaml
model:
  num_classes: 3              # 检测类别数
  pretrained: ./weights/xxx.pt  # 预训练权重路径
  model_scale: s              # 模型规模

training:
  epochs: 60                  # 训练轮数
  batch_size: 2               # 批次大小
  image_size: [800, 800]      # 输入图像尺寸
  lr: 0.00008                 # 学习率
  min_lr: 0.00001             # 最小学习率
  weight_decay: 0.001         # 权重衰减
  grad_clip_norm: 10.0        # 梯度裁剪

  optimizer:
    type: AdamW
    betas: [0.9, 0.999]

  scheduler:
    type: CosineAnnealingLR
    warmup_epochs: 8          # 预热轮数
    warmup_lr: 0.00002        # 预热起始学习率

  loss:
    box_loss_weight: 2.0      # 边界框损失权重
    cls_loss_weight: 2.0      # 分类损失权重
    dfl_loss_weight: 1.5      # DFL损失权重
    focal_gamma: 1.5          # Focal Loss gamma
    focal_alpha: 0.25         # Focal Loss alpha

  amp:
    enabled: true             # 混合精度训练

  gradient_accumulation:
    enabled: true
    steps: 4                  # 累积步数 (等效batch=8)

  sampling:                   # 加权采样配置
    enabled: true
    background_weight: 0.2
    power: 1.15
    class_boosts: [1.0, 2.4, 1.8]  # 各类别增强系数

  evaluation:
    period: 5                 # 评估周期
    conf_threshold: 0.25
    map_conf_threshold: 0.05
    iou_threshold: 0.5

output:
  dir: ./outputs/farmland_finetune_v2
  save_period: 5
```

### 数据集配置 (dataset.yaml)

```yaml
path: <数据集根目录>
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: 细微裂纹
  1: 网状裂隙
  2: 深大裂缝
```

---

## 8. 项目运行方式

### 环境搭建

```powershell
# 方式1: 使用自动配置脚本
cd Farmland_Crack_Detection
scripts\setup_env.bat

# 方式2: 手动配置
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 训练

```powershell
# 基础训练
python train.py --config configs/train.yaml

# 道路裂缝预训练
python train.py --config configs/train_road_crack.yaml

# 农田数据微调
python train.py --config configs/train_farmland_finetune.yaml

# 仅验证
python train.py --config configs/train.yaml --validate --weights outputs/xxx/weights/best.pt
```

### 推理

```powershell
# 单图推理
python inference.py --image test.jpg --model weights/best.pt --conf 0.5 --output result.jpg

# 指定数据集配置
python inference.py --image test.jpg --data data/dataset.yaml
```

### 评估

```powershell
# 完整评估
python evaluate.py --weights outputs/xxx/weights/best.pt --data data/dataset.yaml

# 自定义阈值
python evaluate.py --weights best.pt --conf 0.25 --map-conf 0.05 --iou 0.5
```

### GUI

```powershell
# 启动桌面检测界面
python gui/main_window.py
```

### TensorBoard

```powershell
# 监控训练过程
tensorboard --logdir outputs/
```

### 权重转换

```powershell
# 从 ultralytics YOLOv10 官方权重迁移到自定义模型
python convert_weights.py
```

---

## 9. 数据集说明

### 数据来源

| 数据集 | 路径 | 说明 |
|--------|------|------|
| Roboflow农田数据 | `Farmland_Crack_Detection/data/` | 主训练数据，3类标注 |
| China Drone | `China_Drone/` | 原始无人机航拍数据 |
| Filtered China Drone | `filtered_china_drone/` | 过滤后的无人机数据 (XML→YOLO) |
| New Downloads | `data/new_downloads/` | 新下载待标注数据 |

### 数据集划分

| 划分 | 图像数 | 说明 |
|------|--------|------|
| train | ~94 | 训练集 |
| val | ~21 | 验证集 |
| test | ~18 | 测试集 |

### 标签格式

YOLO格式，每行一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

坐标为归一化值 (0~1)，相对于图像宽高。

### 渐进式训练策略

```
Step 1: 道路裂缝预训练 (train_road_crack.yaml)
        └── 使用 filtered_china_drone 数据集
        └── 输出: outputs/road_crack_pretrain_v3/weights/best.pt

Step 2: 农田数据微调 (train_farmland_finetune.yaml)
        └── 加载 Step 1 的 best.pt 作为预训练
        └── 使用 Roboflow 农田数据集
        └── 启用加权采样 (class_boosts)
        └── 输出: outputs/farmland_finetune_v2/weights/best.pt
```

---

> 文档生成时间: 2026-05-06  
> 基于项目代码库当前状态自动分析生成
