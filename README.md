# Farmland Crack Detection System

基于YOLOv10的农田干裂程度识别系统

## 项目简介

本项目是一个基于YOLOv10目标检测算法的农田土壤干裂程度自动识别系统，用于检测和分类农田中的土壤干裂现象。

## 功能特点

- 自动检测农田图像中的土壤干裂区域
- 将干裂程度分为三个等级：轻度、中度、重度
- 提供友好的图形用户界面
- 支持批量图像处理
- 支持结果导出和报告生成

## 项目结构

```
Farmland_Crack_Detection/
├── configs/                     # 配置文件
├── data/                        # 数据目录
├── models/                      # 模型定义
├── utils/                       # 工具函数
├── gui/                         # GUI界面
├── tests/                       # 测试文件
├── scripts/                     # 脚本
├── docs/                        # 文档
├── weights/                     # 模型权重
├── outputs/                     # 输出目录
├── train.py                     # 训练脚本
├── inference.py                 # 推理脚本
└── requirements.txt             # 依赖包
```

## 🚀 快速开始

### 1. 环境配置

```bash
# Windows用户
scripts\setup_env.bat

# Linux/Mac用户
bash scripts/setup_env.sh
```

### 2. 下载YOLOv10权重

**重要：您需要手动下载YOLOv10预训练权重**

📥 **下载地址：** https://github.com/THU-MIG/yolov10/releases/tag/v1.0

**推荐下载（适合4GB显存）：**

- [yolov10n.pt](https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt) (~6MB)

**下载后放置到：** `weights/yolov10n.pt`

### 3. 准备数据集

将您的农田干裂图像放入相应目录：

- 训练集：`data/raw/train/`
- 验证集：`data/raw/val/`
- 测试集：`data/raw/test/`

### 4. 训练模型

```bash
# 使用低显存配置（适配4GB显存）
python train.py --config configs/train.yaml
```

### 5. 运行推理

```bash
# 单张图像推理
python inference.py --image path/to/image.jpg --model weights/best_model.pt

# 启动GUI界面
python gui/main_window.py
```

## 📋 硬件要求

| 组件  | 最低配置            | 您的配置              | 状态      |
| --- | --------------- | ----------------- | ------- |
| CPU | Intel i5 10代    | i7-10750H         | ✅ 满足    |
| GPU | RTX 3060 (12GB) | GTX 1650 Ti (4GB) | ⚠️ 显存不足 |
| 内存  | 16GB            | 16GB              | ✅ 刚好满足  |

## ⚠️ 4GB显存适配说明

由于您的GPU只有4GB显存，系统已自动配置：

- 批次大小：4（而非16）
- 图像尺寸：512×512（而非640×640）
- 启用混合精度训练（AMP）
- 使用梯度累积模拟大批次效果

## 📊 性能目标

| 指标      | 目标值    |
| ------- | ------ |
| mAP@0.5 | ≥ 90%  |
| FPS     | ≥ 30   |
| 推理延迟    | ≤ 33ms |

## 📚 文档

- [权重下载指南](docs/WEIGHTS_DOWNLOAD.md)
- [技术开发文档](技术开发文档.md)

## 🛠️ 开发进度

- [ ] 项目架构搭建
- [ ] 模型代码实现
- [ ] GUI界面开发
- [ ] 配置文件编写
- [ ] 数据集准备
- [ ] 模型训练
- [ ] 性能优化
- [ ] 系统测试

## 👥 作者信息

- **姓名：** 高一峰
- **学号：** 2212100112
- **专业：** 计算机科学与技术
- **指导教师：** 田颖

## 📞 联系方式

如有问题，请联系指导教师或参考项目文档。

---

**注意：** 本项目为毕业设计项目，仅供学术研究和教学使用。