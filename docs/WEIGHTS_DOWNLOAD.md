# YOLOv10 权重下载指南

## 📥 官方下载地址

**GitHub 发布页面：**
```
https://github.com/THU-MIG/yolov10/releases/tag/v1.0
```

## 📦 推荐下载版本

| 权重文件 | 大小 | 适用场景 | 下载链接 |
|---------|------|---------|----------|
| **yolov10n.pt** | ~6MB | Nano版本，最小最快，适合4GB显存 | [直接下载](https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt) |
| yolov10s.pt | ~21MB | Small版本，中等精度 | [直接下载](https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10s.pt) |
| yolov10m.pt | ~54MB | Medium版本，较高精度 | [直接下载](https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10m.pt) |

## 📁 下载后存放位置

将下载的权重文件放入项目目录：
```
Farmland_Crack_Detection/
└── weights/
    └── yolov10n.pt  ← 推荐下载这个
```

## ⚡ 快速下载命令

如果您使用命令行，可以使用以下命令：

```bash
# 下载推荐的nano版本（适合您的4GB显存）
wget https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt -O weights/yolov10n.pt

# 或者使用curl
curl -L https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt -o weights/yolov10n.pt
```

## 🎯 配置建议

由于您的GPU只有4GB显存，**强烈推荐下载 yolov10n.pt**：
- ✅ 文件最小（~6MB）
- ✅ 训练速度最快
- ✅ 显存占用最少
- ✅ 适合您的硬件配置

## 📋 下载步骤

1. 访问 GitHub 发布页面
2. 找到 `yolov10n.pt` 文件
3. 点击下载
4. 将文件移动到 `weights/` 目录

## 🔧 验证下载

下载完成后，可以通过以下方式验证：

```python
import torch

# 加载权重文件
weights = torch.load('weights/yolov10n.pt', map_location='cpu')
print(f"权重文件大小: {len(weights)} 个键")
print(f"模型类型: {type(weights)}")
```