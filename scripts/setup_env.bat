#!/bin/bash

echo "🎯 环境配置脚本"
echo "================"

# 检查Python版本
echo "📋 检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建conda虚拟环境？(y/n) [默认: n]: " create_env
if [[ "$create_env" == "y" ]]; then
    echo "🐍 创建conda虚拟环境..."
    conda create -n farmland_crack python=3.10 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate farmland_crack
fi

# 安装依赖
echo "📦 安装依赖包..."
pip install --upgrade pip
pip install -r requirements.txt

# 检查CUDA
echo "🔍 检查CUDA环境..."
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  未检测到CUDA，将使用CPU')
"

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p data/raw data/processed data/annotations
mkdir -p weights outputs

# 检查权重文件
echo "🔍 检查权重文件..."
if [ -f "weights/yolov10n.pt" ]; then
    echo "✅ 权重文件已存在"
else
    echo "⚠️  未检测到权重文件"
    echo "📥 请下载权重文件到 weights/ 目录"
    echo "📋 下载地址: https://github.com/THU-MIG/yolov10/releases/tag/v1.0"
fi

echo ""
echo "✅ 环境配置完成！"
echo ""
echo "📋 下一步操作："
echo "  1. 下载YOLOv10权重文件到 weights/ 目录"
echo "  2. 准备数据集"
echo "  3. 运行训练: python train.py --config configs/train.yaml"
echo "  4. 启动GUI: python gui/main_window.py"
