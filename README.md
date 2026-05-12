# 农田干裂检测系统 - 核心文件结构

## 📁 当前项目结构

### 核心功能文件

```
Farmland_Crack_Detection/
├── configs/                    # 配置文件
│   ├── model/
│   │   └── yolov10_crack.yaml # YOLOv10模型配置
│   ├── inference.yaml          # 推理配置
│   └── train_farmland_finetune_v*.yaml # v4-v12 各轮次训练配置
├── data/                       # 数据目录
│   ├── dataset.yaml            # 原始农田数据集配置
│   ├── dataset_copypaste.yaml  # Copy-Paste增强后数据集配置
│   ├── dataset_pseudo.yaml     # 伪标签数据集配置(2401张)
│   ├── train/                  # 训练集（原始手工标注）
│   ├── val/                    # 验证集
│   ├── test/                   # 测试集
│   ├── train_copypaste/        # Copy-Paste增强训练集
│   ├── train_patches/          # 滑窗裁剪Patch训练集
│   └── pseudo_china_drone/     # 伪标签数据(2401张 from China_Drone)
├── models/                     # 模型架构
│   ├── backbone/               # 骨干网络
│   ├── head/                   # 检测头
│   ├── neck/                   # 特征融合
│   ├── attention.py            # 注意力机制
│   └── yolov10_crack.py       # YOLOv10完整模型
├── utils/                      # 工具函数
│   ├── data_processing.py      # 数据处理
│   ├── logger.py               # 日志记录
│   ├── metrics.py              # 评估指标
│   ├── roboflow_dataset.py     # Roboflow数据集适配器（含Mosaic+MixUp增强）
│   └── visualization.py        # 可视化工具
├── gui/                        # 图形界面
│   ├── main_window.py          # 主窗口
│   └── widgets/                # 自定义控件
├── tests/                      # 测试文件
│   ├── test_data.py
│   ├── test_gui.py
│   └── test_model.py
└── tools/                      # 辅助工具脚本（已精简）
    ├── copy_paste_augment.py      # Copy-Paste实例级数据增强
    ├── generate_pseudo_labels.py  # 伪标签生成（Self-Training）
    ├── optimize_thresholds.py     # 搜索最佳评估超参数
    └── patch_extractor.py         # 滑窗裁剪大图为重叠Patch
```

### 主程序文件

```
├── train.py                    # 训练主程序（含EMA、Mosaic、MixUp、采样策略）
├── evaluate.py                 # 评估主程序（含TTA、集成推理）
├── inference.py                # 推理主程序
├── convert_weights.py          # 转换官方权重为兼容格式
├── requirements.txt           # 依赖包列表
└── PROJECT_OVERVIEW.md        # 项目文档

### 训练输出

```
├── outputs/
│   ├── farmland_finetune_v4/   # 当前验证集最佳模型(mAP@0.5=0.2276)
│   ├── farmland_finetune_v5/   # 继续训练探索
│   ├── farmland_finetune_v6/   # 均衡策略尝试
│   ├── farmland_finetune_v7/   # EMA短训探索
│   ├── farmland_finetune_v8/   # 找到真正最优EMA权重(测试集mAP=0.2782)
│   ├── farmland_finetune_v9/   # Patch扩增+数据增强尝试
│   ├── farmland_finetune_v10/  # Bug修复后重训
│   ├── farmland_finetune_v11/  # Mosaic 100%+MixUp过拟合尝试(已终止)
│   └── farmland_finetune_v12/  # 修正增强强度+Copy-Paste数据集训练
```

## 🎯 核心功能

### 1. 模型训练 (`train.py`)

- 支持YOLOv10 anchor-free 架构
- EMA（指数移动平均）模型
- Mosaic / MixUp 数据增强
- 梯度累积+混合精度训练
- Focal Loss + CIOU Loss + DFL Loss
- WeightedRandomSampler 类别不平衡采样

### 2. 模型评估 (`evaluate.py`)

- 完整 mAP@0.5 / Precision / Recall 计算
- 支持 TTA（水平翻转集成）
- 支持多模型集成推理
- 自动加载 EMA 最佳权重

### 3. 图形界面 (`gui/main_window.py`)

- PyQt5界面
- 图像加载、实时推理、结果可视化

## 📊 全部训练轮次对比

| 排名 | 轮次 | mAP@0.5(val) | mAP@0.5(test) | Recall | Precision | 关键特征 |
|------|------|-------------|--------------|--------|-----------|----------|
| 🥇 | **v8** | **0.1701** | **0.2782** | **0.2836** | **0.1814** | **实际最佳权重(EMA修复后)** |
| 🥈 | v10 | 0.2388 | 0.2712 | 0.3433 | 0.1977 | 深度bug修复后 |
| 🥉 | v12 | 0.1701 | 0.2108 | 0.2910 | 0.1831 | Copy-Paste增强 |
| 4 | v4 | 0.2276 | - | 0.3582 | 0.1696 | 原最佳v4 |
| 5 | v6 | 0.2223 | - | 0.3358 | 0.2267 | P/R最均衡 |
| 6 | v9 | 0.1551 | 0.2240 | 0.3134 | 0.2500 | Patch扩增 |

**当前最佳模型权重（测试集）**: `outputs/farmland_finetune_v8/weights/best.pt`

## 🧪 数据集状态

| 数据集 | 来源 | 图片数 | 标注框数 | 说明 |
|--------|------|--------|----------|------|
| train | 手工标注 | 94 张 | 322 框 | 原始训练集 |
| val | 手工标注 | 21 张 | 72 框 | 原始验证集 |
| test | 手工标注 | 68 张 | 213 框 | 原始测试集 |
| pseudo_china_drone | 伪标签生成 | 2401 张 | 24390 框 | China_Drone伪标签(置信度>0.5) |
| pseudo_china_drone(高置信) | 伪标签筛选 | 2081 张 | 3358 框 | 置信度>0.7的高质量子集 |
| train_copypaste | 实例增强 | 446 张 | 2803 框 | Copy-Paste增强后 |

## 🛠️ 技术方案历程

### 已尝试的技术手段

1. **EMA权重提取** - 修复评估脚本正确加载EMA权重，v8测试集mAP从0.17→0.278
2. **Loss代码Bug修复** - 修复grid坐标、DFL边界等3个bug，v10训练指标提升
3. **Copy-Paste实例增强** - 提取裂缝实例随机粘贴融合，增加目标多样性
4. **Mosaic/MixUp增强** - 四图拼接+透明度混合，增强泛化（v11过拟合后调整）
5. **Patch滑窗裁剪** - 大图640x640切片扩增训练样本
6. **伪标签Self-Training** - 对2401张China_Drone图片生成伪标签，待训练验证

### 未尝试的备选方案

- 多尺度训练与测试
- 标签平滑优化
- K-Fold交叉验证
- 更深的骨干网络(从s→m/l)
- 测试时增强(TTA)
- 模型集成(Ensemble)

## 💡 项目优势

- **立即可用**：核心功能完整，无需额外配置
- **硬件适配**：专门为4GB显存优化(batch=4,梯度累积)
- **文档齐全**：每个模块都有详细说明
- **扩展性强**：已集成Mosaic/MixUp/Copy-Paste/伪标签等进阶技术

---

**当前项目状态**：已完成12轮迭代训练，测试集最佳mAP@0.5=0.2782(v8)。伪标签数据已准备就绪(2401张)，下一步进行Self-Training冲刺更高指标。
