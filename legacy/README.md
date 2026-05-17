# ⚠️ Legacy - 废弃代码归档

> **重要提示：此目录中的所有代码均为早期实验性代码，已不再使用！**

## 📁 归档内容

### 1. `tests/` - 废弃的测试脚本
- **问题**: 测试基于自定义的 `YOLOv10Crack` 模型（该模型从未完成开发）
- **状态**: ❌ 无法运行
- **建议**: 如需测试，请使用 pytest + ultralytics 官方接口

**包含文件:**
- `test_model.py` - 测试自定义模型（不存在）
- `test_data.py` - 测试自定义预处理工具
- `test_gui.py` - 测试废弃的GUI组件

---

### 2. `tools/` - 废弃的工具脚本
- **问题**: 大部分工具基于不存在的自定义模型或已被 ultralytics 内置功能替代
- **状态**: ⚠️ 逻辑可参考但无法直接运行

**包含文件:**
- `real_tta.py` - 自定义TTA实现（ultralytics已内置）
- `evaluate_advanced.py` - 高级评估脚本（依赖自定义模型）
- `prepare_dataset_v16.py` - 数据集优化工具（⚠️ 逻辑可参考）
- `generate_visualization_report.py` - 可视化报告生成（部分可用）

---

### 3. `models/` - 废弃的自定义模型
- **问题**: 自定义的 YOLOv10Crack 模型实现，从未集成到主流程
- **原因**: 项目最终采用官方 ultralytics.YOLO 接口

**包含文件:**
- `__init__.py`
- `yolov10_crack.py` - 自定义YOLOv10模型（未完成）
- `attention.py` - CBAM注意力机制模块

---

### 4. `utils/` - 废弃的工具函数
- **问题**: 为支持自定义模型而创建的工具函数，现已被替代

**包含文件:**
- `__init__.py`
- `data_processing.py` - 图像预处理（部分逻辑可参考）
- `roboflow_dataset.py` - 数据集加载器（ultralytics已内置）

---

### 5. `evaluate.py` - 废弃的评估函数
- **问题**: 基于自定义模型的mAP评估函数
- **替代**: 使用 `model.val()` 进行官方评估

---

## 🎯 当前有效的工作流程

### ✅ 训练流程
```bash
cd Farmland_Crack_Detection/scripts
python train_extreme.py
```

### ✅ GUI界面（已修复）
```bash
cd Farmland_Crack_Detection/gui
python main_window.py
```
- **位置**: `/gui/main_window.py`
- **技术栈**: PyQt5 + ultralytics YOLO
- **功能**: 
  - 加载训练好的模型（自动扫描 runs/ 目录）
  - 单张/批量图像检测
  - 三分类可视化标注（细微裂纹/网状裂隙/深大裂缝）
  - 导出检测结果（图像/CSV）

### ✅ 评估方法
```python
from ultralytics import YOLO

model = YOLO('runs/extreme_m/weights/best.pt')
results = model.val()  # 自动评估
```

---

## 📅 归档时间
**归档日期**: 2026-05-16  
**归档原因**: 清理项目结构，移除无价值的实验性代码  
**归档操作者**: AI Assistant  

---

## 💡 如果将来需要类似功能

1. **GUI界面**: 参考 `/gui/main_window.py`（已适配ultralytics）
2. **数据增强**: 参考legacy中的 `prepare_dataset_v16.py` 逻辑
3. **高级评估**: 直接使用 ultralytics 的 TTA 和 val() 功能
4. **自定义模型**: 如果确实需要，建议基于 ultralytics 扩展而非从零编写

---

## ⚡ 快速清理命令

如需彻底删除此目录：
```bash
rm -rf legacy/
```

**警告**: 删除前请确认不需要参考其中的任何代码逻辑！
