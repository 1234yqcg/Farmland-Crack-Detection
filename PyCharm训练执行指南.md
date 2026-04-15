# PyCharm模型训练执行指南

## 📋 目录

1. [项目概述](#项目概述)
2. [环境准备](#环境准备)
3. [PyCharm配置](#pycharm配置)
4. [依赖安装](#依赖安装)
5. [训练执行](#训练执行)
6. [监控与调试](#监控与调试)
7. [常见问题](#常见问题)
8. [训练完成后的操作](#训练完成后的操作)

---

## 项目概述

### 项目信息
- **项目名称**: 基于YOLOv10的农田干裂程度识别系统
- **项目路径**: `d:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection`
- **模型**: YOLOv10n (针对4GB显存优化)
- **任务**: 12类目标检测（包含Drought等类别）

### 数据集信息
- **训练集**: 149张图像
- **验证集**: 41张图像
- **测试集**: 21张图像
- **类别数量**: 12个

---

## 环境准备

### 1. 硬件要求
- **CPU**: Intel i7-10750H ✅
- **GPU**: GTX 1650 Ti (4GB) ⚠️ 需要优化配置
- **内存**: 16GB ✅
- **存储**: 至少10GB可用空间

### 2. 软件要求
- **操作系统**: Windows 10/11
- **Python版本**: 3.11.0 (系统Python)
- **PyCharm版本**: 2023.x 或更新版本
- **CUDA版本**: 11.8 或 12.x

### 3. 文件结构检查
确保以下文件和目录存在：
```
farmland_crack_detection/
├── train.py                      # 训练脚本
├── configs/
│   └── train.yaml                # 训练配置
├── data/
│   ├── dataset.yaml              # 数据集配置
│   ├── train/                    # 训练数据
│   │   ├── images/
│   │   └── labels/
│   ├── val/                      # 验证数据
│   │   ├── images/
│   │   └── labels/
│   └── test/                     # 测试数据
│       ├── images/
│       └── labels/
├── weights/
│   └── yolov10n.pt              # 预训练权重
└── requirements.txt             # 依赖列表
```

---

## Python 3.11.0 配置说明

### 为什么使用系统Python 3.11.0？

**优势：**
- ✅ **性能提升** - Python 3.11比3.9/3.10快10-60%
- ✅ **简单快捷** - 无需创建和管理虚拟环境
- ✅ **节省空间** - 不重复安装Python
- ✅ **易于维护** - 所有包统一管理
- ✅ **完全兼容** - 所有项目依赖都支持Python 3.11

**兼容性验证：**

| 库名 | 版本要求 | Python 3.11支持 | 状态 |
|------|---------|----------------|------|
| PyTorch | ≥2.0.0 | ✅ 完全支持 | ✅ |
| Ultralytics | ≥8.0.0 | ✅ 完全支持 | ✅ |
| OpenCV | ≥4.8.0 | ✅ 完全支持 | ✅ |
| NumPy | ≥1.24.0 | ✅ 完全支持 | ✅ |
| TensorBoard | ≥2.14.0 | ✅ 完全支持 | ✅ |
| PyQt5 | ≥5.15.0 | ✅ 完全支持 | ✅ |

### 系统Python路径

- **Python可执行文件**: `D:\Program Files\Python\Python311\python.exe`
- **版本**: Python 3.11.0
- **安装方式**: 系统全局安装

### 快速配置步骤

1. **验证Python版本**
   ```bash
   python --version
   # 输出: Python 3.11.0
   ```

2. **在PyCharm中配置**
   - `File` → `Settings` → `Python Interpreter`
   - 选择 `Existing environment`
   - 选择 `D:\Program Files\Python\Python311\python.exe`

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 注意事项

⚠️ **如果将来需要多个项目使用不同版本的包：**
- 可以考虑创建虚拟环境
- 或者使用不同的Python版本

但对于当前项目，直接使用系统Python 3.11.0是最佳选择！

---

## PyCharm配置

### 步骤1: 打开项目

1. 启动PyCharm
2. 选择 `File` → `Open`
3. 导航到 `d:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection`
4. 点击 `OK`
5. 等待项目索引完成（右下角进度条）

### 步骤2: 配置Python解释器

1. 打开 `File` → `Settings` (快捷键: `Ctrl+Alt+S`)
2. 导航到 `Project: Farmland_Crack_Detection` → `Python Interpreter`
3. 点击齿轮图标 → `Add Interpreter` → `Add Local Interpreter`
4. 选择 `Existing environment`
5. 点击 `...` 浏览按钮，选择系统Python：
   - **Interpreter**: `D:\Program Files\Python\Python311\python.exe`
6. 点击 `OK` 完成配置

### 步骤3: 验证环境

在PyCharm Terminal中运行：
```bash
python --version
# 应该显示: Python 3.11.0

pip --version
# 应该显示pip版本信息
```

---

## 依赖安装

### 步骤1: 升级pip

在PyCharm Terminal中运行：
```bash
python -m pip install --upgrade pip
```

### 步骤2: 安装PyTorch（CUDA版本）

```bash
# 安装PyTorch 2.0 with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步骤3: 安装项目依赖

```bash
# 方式1: 一次性安装所有依赖
pip install -r requirements.txt

# 方式2: 如果遇到问题，逐个安装核心依赖
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install tensorboard>=2.14.0
pip install tqdm>=4.66.0
pip install PyYAML>=6.0
pip install albumentations>=1.3.0
pip install pycocotools>=2.0.6
```

### 步骤4: 验证安装

```bash
# 验证PyTorch
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

# 验证其他依赖
python -c "import cv2; print('OpenCV版本:', cv2.__version__)"
python -c "import ultralytics; print('Ultralytics版本:', ultralytics.__version__)"
```

**预期输出示例：**
```
PyTorch版本: 2.0.1+cu118
CUDA可用: True
GPU设备: NVIDIA GeForce GTX 1650 Ti
OpenCV版本: 4.8.1.78
Ultralytics版本: 8.0.196
```

---

## 训练执行

### 步骤1: 检查配置文件

#### 检查训练配置 (`configs/train.yaml`)

打开 `configs/train.yaml`，确认以下关键参数：

```yaml
data:
  train_dir: ./data/train/images
  val_dir: ./data/val/images
  annotation_dir: ./data

model:
  num_classes: 12
  pretrained: ./weights/yolov10n.pt
  model_scale: n

training:
  epochs: 300
  batch_size: 4
  image_size: [512, 512]
  lr: 0.001
  min_lr: 0.00001
  weight_decay: 0.0005
  num_workers: 4
  
  amp:
    enabled: true
    
  gradient_accumulation:
    steps: 2
    
  loss:
    box_loss_weight: 7.5
    cls_loss_weight: 1.0
    dfl_loss_weight: 1.5

output:
  dir: ./output
  save_freq: 10
```

#### 检查数据集配置 (`data/dataset.yaml`)

```yaml
path: ./data
train: train/images
val: val/images
test: test/images

nc: 12
names:
  0: Blizzard
  1: Clouds
  2: Drought
  3: Earthquake_damage
  4: Human
  5: Land_slide
  6: Lava_flow
  7: Plane
  8: Sandstorm
  9: Vehicle
  10: Water_disaster
  11: Wildfire
```

### 步骤2: 创建运行配置

1. 点击PyCharm右上角运行配置下拉菜单
2. 选择 `Edit Configurations...`
3. 点击 `+` 号 → `Python`
4. 配置如下：

| 配置项 | 值 |
|--------|-----|
| **Name** | `Train YOLOv10` |
| **Script path** | 点击文件夹图标选择 `train.py` |
| **Parameters** | `--config configs/train.yaml` |
| **Working directory** | `d:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection` |
| **Python interpreter** | 选择系统Python 3.11.0 (`D:\Program Files\Python\Python311\python.exe`) |
| **Environment variables** | `CUDA_VISIBLE_DEVICES=0` (可选) |

5. 点击 `OK` 保存配置

### 步骤3: 测试运行（可选）

在正式训练前，建议先进行测试运行：

1. 修改 `configs/train.yaml` 中的 `epochs` 为 `5`
2. 运行训练，确保流程正常
3. 如果测试成功，将 `epochs` 改回 `300`

### 步骤4: 开始训练

1. 在右上角选择 `Train YOLOv10` 配置
2. 点击绿色运行按钮 ▶️ (或按 `Shift+F10`)
3. 观察底部 `Run` 窗口的输出

**预期训练输出：**
```
📊 train集初始化完成:
  📸 图像数量: 149
  🏷️  标注数量: 149
  📋 类别: ['Blizzard', 'Clouds', 'Drought', ...]
  📦 总标注框: XXX
  🎯 平均每图标注: X.XX

📊 val集初始化完成:
  📸 图像数量: 41
  🏷️  标注数量: 41
  📋 类别: ['Blizzard', 'Clouds', 'Drought', ...]
  📦 总标注框: XXX
  🎯 平均每图标注: X.XX

训练集样本数: 149
验证集样本数: 41
Model parameters: X,XXX,XXX
Mixed precision training enabled
Starting training...
Device: cuda
Epochs: 300
Batch size: 4

Epoch 1: 100%|██████████| 38/38 [00:30<00:00,  1.26it/s, loss=2.3456]
Epoch 2: 100%|██████████| 38/38 [00:28<00:00,  1.34it/s, loss=1.9876]
...
```

---

## 监控与调试

### 1. TensorBoard监控

#### 启动TensorBoard

在PyCharm Terminal中运行：
```bash
tensorboard --logdir output/logs --port 6006
```

#### 查看监控数据

1. 打开浏览器访问: `http://localhost:6006`
2. 查看以下指标：
   - **Loss/train**: 训练损失曲线
   - **Loss/val**: 验证损失曲线
   - **Learning Rate**: 学习率变化
   - **mAP**: 平均精度指标

### 2. GPU监控

在Terminal中运行：
```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 或使用watch命令（需要Git Bash或WSL）
watch -n 1 nvidia-smi
```

**关键指标：**
- **GPU-Util**: GPU利用率（目标：>80%）
- **Memory-Usage**: 显存使用（目标：<3.5GB）
- **Temperature**: GPU温度（目标：<85°C）

### 3. 调试模式

如果遇到问题，使用调试模式：

1. 在 `train.py` 中设置断点（点击行号右侧）
2. 点击调试按钮 🐛 (或按 `Shift+F9`)
3. 逐步执行代码，检查变量值

**常用断点位置：**
- 数据加载后：检查图像和标签格式
- 模型前向传播后：检查输出形状
- 损失计算后：检查损失值

### 4. 日志分析

训练日志保存在 `output/logs/` 目录中：
```bash
# 查看训练日志
type output\logs\training.log

# 或使用PowerShell
Get-Content output\logs\training.log -Tail 50
```

---

## 常见问题

### 问题1: 显存不足 (CUDA out of memory)

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解决方案：**
1. 减小批次大小：
   ```yaml
   # configs/train.yaml
   training:
     batch_size: 2  # 从4改为2
   ```

2. 减小图像尺寸：
   ```yaml
   training:
     image_size: [416, 416]  # 从[512,512]改为[416,416]
   ```

3. 确保混合精度训练启用：
   ```yaml
   training:
     amp:
       enabled: true
   ```

### 问题2: CUDA不可用

**症状：**
```
torch.cuda.is_available() 返回 False
```

**解决方案：**
1. 检查NVIDIA驱动是否安装：
   ```bash
   nvidia-smi
   ```

2. 重新安装CUDA版本的PyTorch：
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. 检查CUDA版本匹配：
   ```bash
   nvcc --version
   ```

### 问题3: 数据加载错误

**症状：**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案：**
1. 检查数据集路径：
   ```bash
   # 确认文件存在
   dir data\train\images | find /c ".jpg"
   dir data\train\labels | find /c ".txt"
   ```

2. 检查dataset.yaml配置：
   ```yaml
   path: ./data  # 确保路径正确
   ```

3. 验证图像和标签对应：
   ```bash
   # 检查图像和标签文件名是否匹配
   dir /b data\train\images > images.txt
   dir /b data\train\labels > labels.txt
   # 对比两个文件
   ```

### 问题4: 训练不收敛

**症状：**
- 损失值不下降或震荡
- 验证损失持续上升

**解决方案：**
1. 调整学习率：
   ```yaml
   training:
     lr: 0.0001  # 降低学习率
   ```

2. 检查数据质量：
   - 确认标注正确
   - 检查图像质量
   - 验证类别平衡

3. 使用学习率预热：
   ```yaml
   training:
     warmup_epochs: 5
     warmup_lr: 0.0001
   ```

### 问题5: 权重文件未找到

**症状：**
```
FileNotFoundError: [Errno 2] No such file or directory: './weights/yolov10n.pt'
```

**解决方案：**
1. 下载YOLOv10权重：
   - 访问: https://github.com/THU-MIG/yolov10/releases/tag/v1.0
   - 下载 `yolov10n.pt`
   - 放置到 `weights/` 目录

2. 检查权重文件：
   ```bash
   dir weights\yolov10n.pt
   ```

3. 更新配置路径：
   ```yaml
   model:
     pretrained: ./weights/yolov10n.pt  # 确保路径正确
   ```

---

## 训练完成后的操作

### 1. 检查训练结果

训练完成后，检查输出目录：
```bash
# 查看输出文件
dir output\weights

# 预期输出：
# best_model.pt          # 最佳模型
# checkpoint_epoch_*.pt   # 各epoch检查点
```

### 2. 评估模型性能

创建评估脚本 `evaluate.py`：
```python
import torch
from models.yolov10_crack import YOLOv10Crack
from utils.roboflow_dataset import RoboflowFarmlandDataset
from torch.utils.data import DataLoader
import yaml

# 加载配置
with open('configs/train.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载模型
model = YOLOv10Crack(num_classes=config['model']['num_classes'])
checkpoint = torch.load('output/weights/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载验证集
val_dataset = RoboflowFarmlandDataset(
    data_yaml_path='data/dataset.yaml',
    split='val',
    image_size=(512, 512),
    augment=False
)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 评估
print(f"最佳损失: {checkpoint['loss']:.4f}")
print(f"训练轮数: {checkpoint['epoch']}")
```

### 3. 模型推理测试

创建推理脚本 `test_inference.py`：
```python
import torch
from models.yolov10_crack import YOLOv10Crack
import cv2
import yaml

# 加载配置
with open('configs/train.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载模型
model = YOLOv10Crack(num_classes=config['model']['num_classes'])
checkpoint = torch.load('output/weights/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载类别名称
with open('data/dataset.yaml', 'r') as f:
    dataset_config = yaml.safe_load(f)
class_names = dataset_config['names']

# 测试推理
image_path = 'data/test/images/test_image.jpg'
image = cv2.imread(image_path)

# 这里添加推理代码
# ...

print("推理测试完成！")
```

### 4. 导出模型

如果需要部署，导出模型：
```python
import torch

# 加载模型
checkpoint = torch.load('output/weights/best_model.pt')

# 导出为ONNX格式（可选）
# torch.onnx.export(model, dummy_input, "model.onnx")

# 导出为TorchScript格式（可选）
# scripted_model = torch.jit.script(model)
# scripted_model.save("model_scripted.pt")

print("模型导出完成！")
```

### 5. 生成训练报告

创建训练报告 `training_report.md`：
```markdown
# 训练报告

## 训练配置
- 模型: YOLOv10n
- 类别数: 12
- 训练轮数: 300
- 批次大小: 4
- 图像尺寸: 512×512
- 学习率: 0.001

## 训练结果
- 最佳损失: X.XXXX
- 最佳轮数: XXX
- 训练时间: XX小时XX分钟

## 性能指标
- mAP@0.5: XX.X%
- 精确率: XX.X%
- 召回率: XX.X%

## 模型文件
- 最佳模型: `output/weights/best_model.pt`
- 模型大小: XX MB
```

---

## 附录

### A. 快速命令参考

```bash
# 环境验证
python --version                         # 查看Python版本
python -c "import torch; print(torch.cuda.is_available())"  # 检查CUDA

# 依赖管理
pip install -r requirements.txt           # 安装依赖
pip list                                 # 查看已安装包
pip install --upgrade pip                 # 升级pip

# 训练相关
python train.py --config configs/train.yaml  # 开始训练
tensorboard --logdir output/logs          # 启动TensorBoard
nvidia-smi                               # 查看GPU状态

# 文件操作
dir data\train\images                   # 列出训练图像
dir output\weights                      # 列出模型权重
```

### B. 配置文件模板

#### 训练配置模板
```yaml
data:
  train_dir: ./data/train/images
  val_dir: ./data/val/images
  annotation_dir: ./data

model:
  num_classes: 12
  pretrained: ./weights/yolov10n.pt
  model_scale: n

training:
  epochs: 300
  batch_size: 4
  image_size: [512, 512]
  lr: 0.001
  min_lr: 0.00001
  weight_decay: 0.0005
  num_workers: 4
  
  amp:
    enabled: true
    
  gradient_accumulation:
    steps: 2
    
  loss:
    box_loss_weight: 7.5
    cls_loss_weight: 1.0
    dfl_loss_weight: 1.5

output:
  dir: ./output
  save_freq: 10
```

### C. 性能优化建议

1. **针对4GB显存优化**
   - 使用混合精度训练 (AMP)
   - 减小批次大小
   - 使用梯度累积
   - 减小图像尺寸

2. **训练速度优化**
   - 增加 `num_workers` (不超过CPU核心数)
   - 使用SSD存储数据集
   - 启用数据缓存

3. **模型性能优化**
   - 使用数据增强
   - 调整学习率调度
   - 使用早停机制

### D. Python 3.11.0 系统配置

#### 配置优势

使用系统Python 3.11.0的优势：
- **性能提升**: Python 3.11比3.9/3.10快10-60%
- **简单快捷**: 无需创建和管理虚拟环境
- **节省空间**: 不重复安装Python
- **易于维护**: 所有包统一管理

#### 完整配置流程

1. **验证系统Python**
   ```bash
   python --version
   # 应该输出: Python 3.11.0
   ```

2. **PyCharm配置**
   - 打开 `File` → `Settings` → `Python Interpreter`
   - 点击 `Add Interpreter` → `Existing environment`
   - 选择 `D:\Program Files\Python\Python311\python.exe`

3. **安装依赖**
   ```bash
   # 升级pip
   python -m pip install --upgrade pip
   
   # 安装PyTorch (CUDA版本)
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # 安装项目依赖
   python -m pip install -r requirements.txt
   ```

4. **验证安装**
   ```bash
   # 检查PyTorch和CUDA
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   
   # 检查GPU
   python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
   ```

#### 常见问题

**Q: 系统Python会影响其他项目吗？**
A: 不会，PyCharm可以为每个项目配置不同的解释器。

**Q: 如何回退到虚拟环境？**
A: 可以随时在PyCharm设置中切换回虚拟环境。

**Q: Python 3.11.0兼容所有依赖吗？**
A: 是的，所有项目依赖都完全支持Python 3.11。

### E. 联系与支持

如遇到问题，请：
1. 检查本文档的常见问题部分
2. 查看PyTorch和YOLOv10官方文档
3. 联系指导教师：田颖

---

**文档版本**: 1.0  
**最后更新**: 2026-03-10  
**适用项目**: 基于YOLOv10的农田干裂程度识别系统