# Windows系统Roboflow数据集操作指南

## 🚨 解决bash命令不可用问题

在Windows系统中，您需要使用PowerShell或CMD，而不是bash。

## 🚀 立即开始（Windows系统）

### 步骤1：使用PowerShell（推荐）
```powershell
# 打开PowerShell，进入项目目录
cd "D:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection"

# 运行PowerShell设置脚本
powershell -ExecutionPolicy Bypass -File ".\scripts\setup_roboflow_env.ps1"
```

### 步骤2：使用简单下载脚本
```powershell
# 使用我们创建的简单脚本
python download_roboflow_simple.py
```

### 步骤3：验证数据集
```powershell
python validate_dataset.py
```

## 📋 Windows命令对照表

| Linux/Mac (bash) | Windows (PowerShell) | Windows (CMD) |
|-------------------|----------------------|---------------|
| `ls` | `Get-ChildItem` 或 `dir` | `dir` |
| `cd path` | `cd path` | `cd path` |
| `python script.py` | `python script.py` | `python script.py` |
| `bash script.sh` | `powershell script.ps1` | 不支持 |
| `mkdir dir` | `New-Item -ItemType Directory dir` | `mkdir dir` |
| `rm file` | `Remove-Item file` | `del file` |

## 🔧 解决PowerShell执行策略问题

### 方法1：临时绕过执行策略（推荐）
```powershell
# 以管理员身份运行PowerShell
powershell -ExecutionPolicy Bypass -File ".\scripts\setup_roboflow_env.ps1"
```

### 方法2：永久修改执行策略
```powershell
# 以管理员身份运行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 方法3：使用CMD运行Python脚本
```cmd
# 使用命令提示符
cd "D:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection"
python download_roboflow_simple.py
```

## 📁 手动操作指南（无需脚本）

### 步骤1：创建目录结构
```powershell
# 创建数据目录
New-Item -ItemType Directory -Path "data\roboflow" -Force

# 检查目录是否创建成功
Get-ChildItem data
```

### 步骤2：从Roboflow网站下载
1. 打开浏览器访问：https://app.roboflow.com/
2. 登录您的账户
3. 找到您的农田干裂检测项目
4. 点击 "Download" 按钮
5. 选择格式：YOLOv8
6. 下载ZIP文件

### 步骤3：解压文件
```powershell
# 假设下载到了Downloads文件夹
$zipFile = "$env:USERPROFILE\Downloads\your_dataset.zip"

# 解压到项目目录
Expand-Archive -Path $zipFile -DestinationPath "data\roboflow"

# 验证解压结果
Get-ChildItem "data\roboflow" -Recurse
```

## 🎯 验证数据集（手动）

### 检查文件结构
```powershell
# 查看数据集目录结构
Get-ChildItem "data\roboflow" -Recurse

# 应该看到：
# data.yaml
# train/images/
# train/labels/
# valid/images/
# valid/labels/
# test/images/
# test/labels/
```

### 统计文件数量
```powershell
# 统计训练集
$trainImages = Get-ChildItem "data\roboflow\train\images" -Include *.jpg,*.png -Recurse
$trainLabels = Get-ChildItem "data\roboflow\train\labels" -Include *.txt -Recurse
Write-Host "训练集: $($trainImages.Count) 图像, $($trainLabels.Count) 标签"

# 统计验证集
$valImages = Get-ChildItem "data\roboflow\valid\images" -Include *.jpg,*.png -Recurse
$valLabels = Get-ChildItem "data\roboflow\valid\labels" -Include *.txt -Recurse
Write-Host "验证集: $($valImages.Count) 图像, $($valLabels.Count) 标签"
```

### 检查data.yaml文件
```powershell
# 查看配置文件内容
Get-Content "data\roboflow\data.yaml"
```

## 🚀 快速命令序列

### 完整操作流程
```powershell
# 1. 进入项目目录
cd "D:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection"

# 2. 创建数据目录
New-Item -ItemType Directory -Path "data\roboflow" -Force

# 3. 使用简单下载脚本（如果已创建）
python download_roboflow_simple.py

# 4. 验证数据集
python validate_dataset.py

# 5. 下载YOLOv10权重（手动操作）
# 访问: https://github.com/THU-MIG/yolov10/releases/tag/v1.0
# 下载 yolov10n.pt 到 weights/ 目录

# 6. 开始训练
python train.py --config configs/train.yaml
```

## 🛠️ 故障排除

### 问题1：Python命令找不到
```powershell
# 检查Python安装
where python

# 如果找不到，手动指定路径
C:\Python310\python.exe download_roboflow_simple.py
```

### 问题2：PowerShell无法识别脚本
```powershell
# 使用完整路径
& "C:\Python310\python.exe" download_roboflow_simple.py
```

### 问题3：文件路径问题
```powershell
# 使用绝对路径
$projectPath = "D:\桌面\学习\000毕业设计！\YOLOv10\Farmland_Crack_Detection"
cd $projectPath
```

### 问题4：网络连接问题
```powershell
# 检查网络连接
Test-Connection -ComputerName google.com -Count 4

# 如果使用代理
$env:HTTP_PROXY="http://your-proxy:port"
$env:HTTPS_PROXY="http://your-proxy:port"
```

## 📊 成功验证指标

✅ **数据集结构正确**：
- data.yaml 文件存在且格式正确
- train/valid/test 三个目录都存在
- 每个目录下都有 images/ 和 labels/ 子目录

✅ **文件数量合理**：
- 训练集：建议 >200 张图像
- 验证集：建议 >50 张图像
- 测试集：建议 >50 张图像

✅ **类别平衡**：
- 轻度干裂：mild 类别
- 中度干裂：moderate 类别
- 重度干裂：severe 类别

## 🎯 下一步操作

1. **获取Roboflow API密钥**：
   - 访问 https://app.roboflow.com/
   - 登录 → Settings → API Keys
   - 复制您的API密钥

2. **获取项目信息**：
   - 在Roboflow中找到您的项目
   - 记录工作区名称和项目名称
   - 查看项目URL中的信息

3. **下载数据集**：
   - 使用脚本自动下载，或
   - 手动从网站下载

4. **验证数据集**：
   - 运行验证脚本
   - 检查文件完整性

5. **开始训练**：
   - 下载YOLOv10权重
   - 配置训练参数
   - 运行训练脚本

## 💡 温馨提示

- **Windows系统**不支持bash命令，请使用PowerShell或CMD
- **PowerShell脚本**需要管理员权限运行
- **Python脚本**可以在任何命令行中运行
- **手动下载**是最简单的方法，适合初学者
- **API下载**适合自动化和批量操作

如果您需要我帮您获取具体的Roboflow项目信息或API密钥，请告诉我您的项目详情！