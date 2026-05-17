# -*- coding: utf-8 -*-
"""
农田干裂程度识别系统 - Windows兼容版本
彻底解决OpenCV/ultralytics与PyQt5的冲突
"""

import sys
import os

# ============================================================
# 第1步：设置环境变量（必须在所有导入之前）
# ============================================================

_pyqt5_install_path = None

# Windows和Linux兼容的PyQt5安装路径检测
_possible_paths = [
    os.path.join(os.environ.get('TEMP', '/tmp'), 'pyqt5_install'),  # Windows TEMP or Linux /tmp
    r'C:\pyqt5_install',  # Windows特定路径
    '/tmp/pyqt5_install',  # Linux路径
]

for _path in _possible_paths:
    if os.path.exists(_path):
        _pyqt5_install_path = _path
        break

if _pyqt5_install_path:
    sys.path.insert(0, _pyqt5_install_path)
    
    _qt_plugins = os.path.join(_pyqt5_install_path, 'PyQt5', 'Qt5', 'plugins')
    if os.path.exists(_qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = _qt_plugins
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(_qt_plugins, 'platforms')
    
    # 根据操作系统选择平台
    if sys.platform == 'win32':
        os.environ['QT_QPA_PLATFORM'] = 'windows'
    else:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Linux
    
    os.environ['OPENCV_GUI_DISABLE'] = 'True'

# ============================================================
# 第2步：只导入PyQt5（安全的）
# ============================================================

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar,
    QProgressBar, QTextEdit, QGroupBox, QDoubleSpinBox,
    QMessageBox, QComboBox, QApplication, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage


# ============================================================
# 第3步：延迟导入其他库（在需要时才导入）
# ============================================================

def get_cv2():
    """延迟导入cv2"""
    import cv2
    return cv2

def get_numpy():
    """延迟导入numpy"""
    import numpy as np
    return np

def get_pil_image():
    """延迟导入PIL"""
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont

def get_yolo():
    """延迟导入ultralytics YOLO"""
    from ultralytics import YOLO
    return YOLO


def draw_chinese_text(img, text, position, font_size=20, color=(255, 255, 255), bg_color=None):
    """
    在OpenCV图像上绘制中文文字（使用PIL解决中文编码问题）
    
    Args:
        img: OpenCV图像 (BGR格式)
        text: 要绘制的文本
        position: (x, y) 文字位置
        font_size: 字体大小
        color: 文字颜色 (B, G, R)
        bg_color: 背景颜色 (B, R, G)，可选
    """
    Image, ImageDraw, ImageFont = get_pil_image()
    np = get_numpy()
    cv2 = get_cv2()
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体（Windows优先）
    try:
        # 根据操作系统选择字体路径
        if sys.platform == 'win32':
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',      # Windows黑体
                'C:/Windows/Fonts/msyh.ttf',         # Windows微软雅黑
                'C:/Windows/Fonts/simsun.ttc',       # Windows宋体
                'C:/Windows/Fonts/simkai.ttf',       # Windows楷体
            ]
        elif sys.platform == 'darwin':  # macOS
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
            ]
        else:  # Linux
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制背景矩形（可选）
    if bg_color:
        bbox = draw.textbbox(position, text, font=font)
        padding = 5
        bg_rect = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ]
        draw.rectangle(bg_rect, fill=bg_color)
    
    # 绘制文字（PIL使用RGB格式）
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class InferenceThread(QThread):
    finished = pyqtSignal(list, object)
    progress = pyqtSignal(int)
    
    def __init__(self, model, image, conf_threshold: float, iou_threshold: float):
        super().__init__()
        self.model = model
        self.image = image
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def run(self):
        try:
            np = get_numpy()
            
            self.progress.emit(30)
            self.progress.emit(60)
            
            results = self.model(
                self.image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            self.progress.emit(90)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes_data = result.boxes.data.cpu().numpy()
                    
                    for box in boxes_data:
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = float(box[4])
                        class_id = int(box[5])
                        
                        class_names = [
                            '细微裂纹', '网状裂隙', '深大裂缝'
                        ]
                        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_name': class_name,
                            'class_id': class_id
                        })
            
            self.progress.emit(100)
            self.finished.emit(detections, self.image)
            
        except Exception as e:
            print(f"Inference error: {e}")
            self.finished.emit([], self.image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("农田干裂程度识别系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.model = None
        self.current_image = None
        self.current_results = None
        self.model_path = None
        self.image_files = []
        self.current_image_index = 0
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.runs_dir = os.path.join(project_root, 'runs')
        
        self._init_ui()
        self._refresh_model_list()
        
        self.model_path_edit.returnPressed.connect(self.on_model_path_enter)
    
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=5)

        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        self._create_menu_bar()
        self._create_status_bar()
    
    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("模型路径:"))
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("输入模型路径或点击浏览选择...")
        self.model_path_edit.setMinimumWidth(250)
        self.model_path_edit.setMaximumWidth(400)
        path_layout.addWidget(self.model_path_edit)
        
        btn_browse = QPushButton("📂 浏览")
        btn_browse.clicked.connect(self.browse_model_file)
        btn_browse.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        path_layout.addWidget(btn_browse)
        
        model_layout.addLayout(path_layout)
        
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("快速选择:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(250)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        combo_layout.addWidget(self.model_combo)
        
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self._refresh_model_list)
        combo_layout.addWidget(self.btn_refresh)
        
        model_layout.addLayout(combo_layout)
        layout.addWidget(model_group)
        
        button_layout = QHBoxLayout()
        
        self.btn_load_image = QPushButton("📷 加载图像")
        self.btn_load_image.clicked.connect(self.load_image)
        button_layout.addWidget(self.btn_load_image)
        
        self.btn_load_folder = QPushButton("📁 批量加载")
        self.btn_load_folder.clicked.connect(self.load_folder)
        button_layout.addWidget(self.btn_load_folder)
        
        self.btn_prev = QPushButton("⬅️ 上一张")
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_prev.setEnabled(False)
        button_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("➡️ 下一张")
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_next.setEnabled(False)
        button_layout.addWidget(self.btn_next)
        
        layout.addLayout(button_layout)
        
        detect_layout = QHBoxLayout()
        self.btn_detect = QPushButton("🔍 开始检测")
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        detect_layout.addStretch()
        detect_layout.addWidget(self.btn_detect)
        detect_layout.addStretch()
        layout.addLayout(detect_layout)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0; 
                border: 2px solid #ddd;
                border-radius: 8px;
            }
        """)
        self.image_label.setText("\n\n请加载图像文件\n\n支持格式：JPG, PNG, BMP")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        return panel
    
    def _create_right_panel(self):
        panel = QWidget()
        # 固定右侧面板宽度，防止布局抖动
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        settings_group = QGroupBox("⚙️ 检测参数")
        settings_layout = QVBoxLayout(settings_group)
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_slider = QDoubleSpinBox()
        self.conf_slider.setRange(0.01, 0.99)
        self.conf_slider.setValue(0.25)
        self.conf_slider.setSingleStep(0.05)
        self.conf_slider.setDecimals(2)
        conf_layout.addWidget(self.conf_slider)
        settings_layout.addLayout(conf_layout)
        
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("NMS阈值:"))
        self.iou_slider = QDoubleSpinBox()
        self.iou_slider.setRange(0.1, 0.9)
        self.iou_slider.setValue(0.45)
        self.iou_slider.setSingleStep(0.05)
        self.iou_slider.setDecimals(2)
        iou_layout.addWidget(self.iou_slider)
        settings_layout.addLayout(iou_layout)
        
        layout.addWidget(settings_group)
        
        results_group = QGroupBox("📊 检测结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        self.results_text.setFontFamily("Consolas")
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        export_group = QGroupBox("💾 导出结果")
        export_layout = QVBoxLayout(export_group)
        
        self.btn_export_image = QPushButton("导出标注图像")
        self.btn_export_image.clicked.connect(self.export_result_image)
        export_layout.addWidget(self.btn_export_image)
        
        self.btn_export_csv = QPushButton("导出CSV报告")
        self.btn_export_csv.clicked.connect(self.export_csv_report)
        export_layout.addWidget(self.btn_export_csv)
        
        layout.addWidget(export_group)
        
        info_group = QGroupBox("ℹ️ 系统信息")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("系统就绪")
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(150)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction("加载图像(&I)", self.load_image, "Ctrl+O")
        file_menu.addAction("批量加载(&B)", self.load_folder)
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, "Ctrl+Q")
        
        model_menu = menubar.addMenu("模型(&M)")
        model_menu.addAction("刷新模型列表", self._refresh_model_list)
        model_menu.addAction("浏览模型文件", self.browse_model_file)
        
        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction("关于(&A)", self.show_about)
    
    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def _refresh_model_list(self):
        self.model_combo.clear()
        self.model_combo.addItem("-- 请选择训练好的模型 --", None)
        
        runs_dirs = []
        if os.path.exists(self.runs_dir):
            for exp_name in sorted(os.listdir(self.runs_dir)):
                exp_path = os.path.join(self.runs_dir, exp_name)
                if os.path.isdir(exp_path):
                    weights_dir = os.path.join(exp_path, 'weights')
                    if os.path.exists(weights_dir):
                        best_model = os.path.join(weights_dir, 'best.pt')
                        last_model = os.path.join(weights_dir, 'last.pt')
                        if os.path.exists(best_model):
                            runs_dirs.append((f"{exp_name} (最佳)", best_model))
                        if os.path.exists(last_model):
                            runs_dirs.append((f"{exp_name} (最新)", last_model))
        
        for display_name, path in runs_dirs:
            self.model_combo.addItem(display_name, path)
    
    def on_model_selected(self, index):
        model_path = self.model_combo.currentData()
        if model_path is None or not os.path.exists(model_path):
            return
        
        self.model_path_edit.setText(model_path)
        self._load_ultralytics_model(model_path)
    
    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重文件",
            self.runs_dir if os.path.exists(self.runs_dir) else "",
            "PyTorch模型 (*.pt);;所有文件 (*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
            self._load_ultralytics_model(file_path)
    
    def on_model_path_enter(self):
        model_path = self.model_path_edit.text().strip()
        if model_path and os.path.exists(model_path):
            self._load_ultralytics_model(model_path)
        elif model_path:
            QMessageBox.warning(self, "警告", f"模型文件不存在:\n{model_path}")
    
    def _load_ultralytics_model(self, model_path: str):
        try:
            YOLO = get_yolo()
            
            self.status_bar.showMessage(f"正在加载模型: {os.path.basename(os.path.dirname(os.path.dirname(model_path)))}")
            self.info_label.setText(f"⏳ 加载中...\n{model_path}")
            
            QApplication.instance().processEvents()
            
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            self.status_bar.showMessage(f"✓ 模型已加载: {exp_name}")
            
            info_text = f"""✅ 模型信息
━━━━━━━━━━━━━━━
实验名称: {exp_name}
权重路径: {model_path}
状态: 已就绪

可进行图像检测"""
            
            self.info_label.setText(info_text)
            
            if self.current_image is not None:
                self.btn_detect.setEnabled(True)
            
            # 强制刷新布局，确保UI一致性
            QApplication.instance().processEvents()
            self.adjustSize()
                
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            QMessageBox.critical(self, "错误", error_msg)
            self.status_bar.showMessage(error_msg)
            self.info_label.setText(f"❌ 加载失败\n{error_msg}")
            self.model = None
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif);;所有文件 (*)"
        )
        
        if file_path:
            try:
                Image, ImageDraw, ImageFont = get_pil_image()
                cv2 = get_cv2()
                
                pil_img = Image.open(file_path).convert('RGB')
                np = get_numpy()
                self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                if self.current_image is not None:
                    self.display_image(self.current_image)
                    
                    if self.model is not None:
                        self.btn_detect.setEnabled(True)
                    
                    self.status_bar.showMessage(f"✓ 已加载: {os.path.basename(file_path)} ({self.current_image.shape[1]}×{self.current_image.shape[0]})")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败: {e}")
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含图像的文件夹")
        
        if folder_path:
            from pathlib import Path
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend([str(f) for f in Path(folder_path).glob(f'*{ext}')])
                image_files.extend([str(f) for f in Path(folder_path).glob(f'*{ext.upper()}')])
            
            image_files = list(set(image_files))
            image_files.sort(key=str.lower)
            
            if image_files:
                self.image_files = image_files
                self.current_image_index = 0
                
                self._load_current_image_from_list()
                
            else:
                QMessageBox.warning(self, "警告", f"文件夹中没有找到图像文件:\n{folder_path}")
    
    def _load_current_image_from_list(self):
        if 0 <= self.current_image_index < len(self.image_files):
            file_path = self.image_files[self.current_image_index]
            
            try:
                Image, ImageDraw, ImageFont = get_pil_image()
                cv2 = get_cv2()
                np = get_numpy()
                
                pil_img = Image.open(file_path).convert('RGB')
                self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                if self.current_image is not None:
                    self.display_image(self.current_image)
                    
                    if self.model is not None:
                        self.btn_detect.setEnabled(True)
                    
                    total_images = len(self.image_files)
                    self.status_bar.showMessage(f"✓ 图像 {self.current_image_index + 1}/{total_images}: {os.path.basename(file_path)}")
                    
                    self.btn_prev.setEnabled(self.current_image_index > 0)
                    self.btn_next.setEnabled(self.current_image_index < total_images - 1)
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败: {e}")
    
    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._load_current_image_from_list()
    
    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self._load_current_image_from_list()
    
    def display_image(self, image):
        qimage = QImage(image.data, image.shape[1], image.shape[0], 
                       image.shape[2] * image.shape[1], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def start_detection(self):
        if self.model is None or self.current_image is None:
            return
        
        self.btn_detect.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        
        self.inference_thread = InferenceThread(
            self.model,
            self.current_image,
            self.conf_slider.value(),
            self.iou_slider.value()
        )
        self.inference_thread.progress.connect(self.update_progress)
        self.inference_thread.finished.connect(self.on_detection_finished)
        self.inference_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def on_detection_finished(self, detections, original_image):
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        
        if detections:
            result_image = self._draw_detections(original_image.copy(), detections)
            self.display_image(result_image)
            self.current_results = {
                'detections': detections,
                'image': result_image
            }
            
            results_str = "检测结果:\n" + "="*50 + "\n"
            for i, det in enumerate(detections, 1):
                bbox = det['bbox']
                results_str += f"\n[{i}] {det['class_name']} ({det['confidence']:.2%})\n"
                results_str += f"    位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n"
            
            results_str += "\n" + "="*50 + f"\n共检测到 {len(detections)} 个目标\n"
            self.results_text.setText(results_str)
            
            self.status_bar.showMessage(f"✓ 检测完成，发现 {len(detections)} 个目标")
            
        else:
            self.display_image(original_image)
            self.results_text.setText("未检测到任何目标\n\n尝试降低置信度阈值")
            self.status_bar.showMessage("未检测到目标")
    
    def _draw_detections(self, image, detections):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (255, 128, 0), (0, 128, 255),
            (128, 255, 0), (255, 0, 128), (0, 255, 128)
        ]
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = colors[class_id % len(colors)]
            
            cv2 = get_cv2()
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            label_y = max(bbox[1], label_size[1] + 10)
            cv2.rectangle(image, 
                         (bbox[0], label_y - label_size[1] - 10),
                         (bbox[0] + label_size[0] + 6, label_y + 4),
                         color, -1)
            
            image = draw_chinese_text(image, label, (bbox[0] + 3, label_y - label_size[1] - 5),
                                     font_size=14, color=(255, 255, 255))
        
        return image
    
    def export_result_image(self):
        if self.current_results is None:
            QMessageBox.warning(self, "警告", "没有可导出的检测结果")
            return
        
        default_name = f"detection_result_{int(time.time())}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存标注图像",
            default_name,
            "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
        )
        
        if save_path:
            import time
            cv2 = get_cv2()
            cv2.imwrite(save_path, self.current_results['image'])
            QMessageBox.information(self, "成功", f"图像已保存到:\n{save_path}")
    
    def export_csv_report(self):
        if self.current_results is None:
            QMessageBox.warning(self, "警告", "没有可导出的检测结果")
            return
        
        import time
        from datetime import datetime
        
        default_name = f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV报告",
            default_name,
            "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if save_path:
            import csv
            with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['序号', '类别ID', '类别名称', '置信度', 'X1', 'Y1', 'X2', 'Y2', '宽度', '高度'])
                
                for i, det in enumerate(self.current_results['detections'], 1):
                    bbox = det['bbox']
                    writer.writerow([
                        i,
                        det['class_id'],
                        det['class_name'],
                        f"{det['confidence']:.4f}",
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ])
            
            QMessageBox.information(self, "成功", f"CSV报告已保存到:\n{save_path}")
    
    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            """
<h2>农田干裂程度识别系统</h2>
<p>基于YOLOv10的深度学习目标检测系统</p>
<p>版本：2.0 (Windows兼容版)</p>
<br>
<p><b>功能特性：</b></p>
<ul>
<li>支持3类农田裂缝检测（细微裂纹/网状裂隙/深大裂缝）</li>
<li>实时图像检测</li>
<li>批量处理</li>
<li>结果可视化与导出</li>
</ul>
<br>
<p>技术栈：Python + PyTorch + Ultralytics YOLOv10 + PyQt5</p>
"""
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())