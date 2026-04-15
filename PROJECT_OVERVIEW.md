# 农田干裂检测系统 - 清理后的核心文件结构

## 📁 当前项目结构

### 核心功能文件

```
Farmland_Crack_Detection/
├── configs/                    # 配置文件
│   ├── model/
│   │   └── yolov10_crack.yaml # YOLOv10模型配置
│   ├── inference.yaml          # 推理配置
│   ├── train.yaml              # 训练配置
│   └── train_personal.yaml     # 个性化训练配置（已优化4GB显存）
├── data/                       # 数据目录
│   └── dataset.yaml            # 数据集配置
├── models/                     # 模型架构
│   ├── backbone/               # 骨干网络
│   ├── head/                   # 检测头
│   ├── neck/                   # 特征融合
│   ├── __init__.py
│   ├── attention.py            # 注意力机制
│   └── yolov10_crack.py       # YOLOv10完整模型
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── data_processing.py      # 数据处理
│   ├── logger.py               # 日志记录
│   ├── metrics.py              # 评估指标
│   ├── roboflow_dataset.py     # Roboflow数据集适配器
│   └── visualization.py        # 可视化工具
├── gui/                        # 图形界面
│   ├── __init__.py
│   ├── main_window.py          # 主窗口
│   └── widgets/                # 自定义控件
├── tests/                      # 测试文件
│   ├── __init__.py
│   ├── test_data.py            # 数据测试
│   ├── test_gui.py             # GUI测试
│   └── test_model.py           # 模型测试
└── tools/                      # 辅助工具脚本
    ├── analyze_dataset.py      # 数据集分析
    ├── batch_test.py           # 批量测试
    ├── clean_dataset.py        # 数据集清洗
    ├── debug_anchor.py         # 锚框调试
    ├── rebuild_dataset.py      # 数据集重建
    ├── split_dataset.py        # 数据集划分
    ├── check_*.py              # 各种检查脚本
    └── test_*.py               # 各种测试脚本
```

### 主程序文件

```
├── train.py                    # 训练主程序
├── inference.py               # 推理主程序
├── analyze_weights.py         # 分析官方权重层名称
├── convert_weights.py         # 转换官方权重为兼容格式
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目说明
└── get_roboflow_info.py      # Roboflow信息获取工具
```

### 权重文件

```
├── weights/
│   ├── yolov10n.pt            # YOLOv10n官方权重
│   ├── yolov10s.pt            # YOLOv10s官方权重
│   └── yolov10s_backbone.pt   # 转换后的兼容权重(222层)
```

### 配置文件

```
├── configs/
│   ├── model/
│   │   └── yolov10_crack.yaml # YOLOv10模型配置
│   ├── inference.yaml          # 推理配置
│   ├── train.yaml              # 训练配置
│   ├── train_personal.yaml     # 个性化训练配置（已优化4GB显存）
│   └── train_road_crack.yaml   # 道路裂缝预训练配置
```

### 公开模型相关文件

```
├── final_api_test.py          # API测试工具
├── download_and_inference_public.py  # 公开模型下载和推理
└── PUBLIC_MODEL_GUIDE.md      # 公开模型使用指南
```

## 🎯 核心功能

### 1. 模型训练 (`train.py`)

- 支持YOLOv10架构
- 适配4GB显存优化配置
- 混合精度训练
- 梯度累积
- 自动模型保存

### 2. 模型推理 (`inference.py`)

- 单张图像推理
- 批量图像处理
- 结果可视化
- 性能评估

### 3. 图形界面 (`gui/main_window.py`)

- PyQt5界面
- 图像加载和显示
- 实时推理
- 结果保存

### 4. 数据处理 (`utils/`)

- 数据增强
- 格式转换
- 验证工具
- 可视化功能

## 📊 项目特点

✅ **针对您的硬件优化**：

- 批次大小：4（适配4GB显存）
- 图像尺寸：512×512
- 混合精度训练启用
- 梯度累积：4步

✅ **完整的工具链**：

- 训练 → 验证 → 测试 → 推理
- 图形界面 → 批量处理 → 结果可视化
- 数据下载 → 预处理 → 增强

✅ **毕业设计友好**：

- 详细的文档和注释
- 模块化设计
- 易于扩展和修改
- 完整的测试覆盖

## 🚀 下一步操作

### 如果您已下载好数据集：

1. **放置数据集**到 `data/roboflow/` 目录
2. **验证数据集**运行数据验证脚本
3. **下载YOLOv10权重**到 `weights/` 目录
4. **开始训练**使用个性化配置

### 如果使用公开模型：

1. **测试API**使用 `final_api_test.py`
2. **体验推理**使用公开模型
3. **集成到GUI**添加到图形界面
4. **对比性能**与自定义模型对比

## 💡 项目优势

- **立即可用**：核心功能完整，无需额外配置
- **硬件适配**：专门为您的4GB显存优化
- **文档齐全**：每个模块都有详细说明
- **扩展性强**：易于添加新功能和改进
- **毕业设计完美**：符合学术项目要求

**您的农田干裂检测系统已经准备就绪！** 🎯

需要我帮您进行下一步操作吗？比如开始训练或测试推理功能？