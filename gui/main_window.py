import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar,
    QProgressBar, QTextEdit, QGroupBox, QDoubleSpinBox,
    QMessageBox, QLineEdit, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov10_crack import YOLOv10Crack

class InferenceThread(QThread):
    finished = pyqtSignal(list, np.ndarray)
    progress = pyqtSignal(int)
    
    def __init__(self, model, image, image_size, conf_threshold, iou_threshold):
        super().__init__()
        self.model = model
        self.image = image
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def run(self):
        self.progress.emit(30)
        
        original_h, original_w = self.image.shape[:2]
        resized = cv2.resize(self.image, (self.image_size, self.image_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, 0)
        
        self.progress.emit(60)
        
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(normalized))
        
        outputs = outputs.squeeze(0).numpy()
        
        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        
        high_conf_mask = scores > self.conf_threshold
        high_conf_boxes = boxes[high_conf_mask]
        high_conf_scores = scores[high_conf_mask]
        
        results = []
        if len(high_conf_boxes) > 0:
            keep = self.nms(high_conf_boxes, high_conf_scores, self.iou_threshold)
            
            scale_x = original_w / self.image_size
            scale_y = original_h / self.image_size
            
            for idx in keep:
                x1, y1, x2, y2 = high_conf_boxes[idx]
                conf = high_conf_scores[idx]
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_name': 'Crack'
                })
        
        self.progress.emit(100)
        self.finished.emit(results, self.image)
    
    def nms(self, boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("农田裂缝检测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.model = None
        self.current_image = None
        self.current_results = None
        self.model_path = None
        self.image_size = 512
        self.image_files = []
        self.current_image_index = 0
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
        
        self._init_ui()
        self._refresh_model_list()
    
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=3)
        
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        self._create_menu_bar()
        self._create_status_bar()
    
    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        model_layout = QHBoxLayout()
        
        #odel_layout.addWidget(QLabel("加载模型"))
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_combo)
        
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self._refresh_model_list)
        model_layout.addWidget(self.btn_refresh)
        
        layout.addLayout(model_layout)
        
        button_layout = QHBoxLayout()
        
        self.btn_load_image = QPushButton("加载图像")
        self.btn_load_image.clicked.connect(self.load_image)
        button_layout.addWidget(self.btn_load_image)
        
        self.btn_load_folder = QPushButton("批量加载")
        self.btn_load_folder.clicked.connect(self.load_folder)
        button_layout.addWidget(self.btn_load_folder)
        
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_prev.setEnabled(False)
        button_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("下一张")
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_next.setEnabled(False)
        button_layout.addWidget(self.btn_next)
        
        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        button_layout.addWidget(self.btn_detect)
        
        layout.addLayout(button_layout)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        layout.addWidget(self.image_label)
        
        return panel
    
    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        settings_group = QGroupBox("检测设置")
        settings_layout = QVBoxLayout(settings_group)
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_slider = QDoubleSpinBox()
        self.conf_slider.setRange(0.01, 0.9)
        self.conf_slider.setValue(0.15)
        self.conf_slider.setSingleStep(0.05)
        conf_layout.addWidget(self.conf_slider)
        settings_layout.addLayout(conf_layout)
        
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU阈值:"))
        self.iou_slider = QDoubleSpinBox()
        self.iou_slider.setRange(0.1, 0.9)
        self.iou_slider.setValue(0.3)
        self.iou_slider.setSingleStep(0.1)
        iou_layout.addWidget(self.iou_slider)
        settings_layout.addLayout(iou_layout)
        
        layout.addWidget(settings_group)
        
        results_group = QGroupBox("检测结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout(export_group)
        
        self.btn_export_image = QPushButton("导出结果图像")
        self.btn_export_image.clicked.connect(self.export_result_image)
        export_layout.addWidget(self.btn_export_image)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("加载图像", self.load_image)
        file_menu.addAction("加载文件夹", self.load_folder)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)
        
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("关于", self.show_about)
    
    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def _refresh_model_list(self):
        self.model_combo.clear()
        self.model_combo.addItem("请选择模型...", None)
        
        if os.path.exists(self.output_dir):
            for exp_name in os.listdir(self.output_dir):
                exp_path = os.path.join(self.output_dir, exp_name)
                if os.path.isdir(exp_path):
                    weights_path = os.path.join(exp_path, 'weights')
                    if os.path.exists(weights_path):
                        best_model = os.path.join(weights_path, 'best.pt')
                        if os.path.exists(best_model):
                            self.model_combo.addItem(f"{exp_name}/best.pt", best_model)
        
        if self.model_combo.count() > 1:
            pass
    
    def on_model_selected(self, index):
        model_path = self.model_combo.currentData()
        if model_path is None:
            return
        
        try:
            self.status_bar.showMessage("正在加载模型...")
            
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_state = checkpoint.get('model', checkpoint)
            
            self.model = YOLOv10Crack(
                num_classes=1,
                depth_multiple=0.33,
                width_multiple=0.50,
                use_attention=True,
                reg_max=16
            )
            
            model_dict = self.model.state_dict()
            filtered_state = {}
            for k, v in model_state.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_state[k] = v
            
            self.model.load_state_dict(filtered_state, strict=False)
            self.model.eval()
            
            self.model_path = model_path
            self.status_bar.showMessage(f"模型已加载: {os.path.basename(os.path.dirname(model_path))}")
            
            if self.current_image is not None:
                self.btn_detect.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {e}")
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.pt *.pth);;所有文件 (*)"
        )
        
        if file_path:
            try:
                self.status_bar.showMessage("正在加载模型...")
                
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                model_state = checkpoint.get('model', checkpoint)
                
                self.model = YOLOv10Crack(
                    num_classes=1,
                    depth_multiple=0.33,
                    width_multiple=0.50,
                    use_attention=True,
                    reg_max=16
                )
                
                model_dict = self.model.state_dict()
                filtered_state = {}
                for k, v in model_state.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        filtered_state[k] = v
                
                self.model.load_state_dict(filtered_state, strict=False)
                self.model.eval()
                
                self.model_path = file_path
                self.model_label.setText(os.path.basename(file_path))
                self.model_label.setStyleSheet("color: green;")
                self.status_bar.showMessage(f"模型已加载: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {e}")
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_path:
            try:
                pil_img = Image.open(file_path).convert('RGB')
                self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                if self.current_image is not None:
                    self.display_image(self.current_image)
                    self.btn_detect.setEnabled(True)
                    self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败: {e}")
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        
        if folder_path:
            from pathlib import Path
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                for f in Path(folder_path).glob(f'*{ext}'):
                    image_files.append(str(f))
            
            image_files = list(set(image_files))
            image_files.sort()
            
            if image_files:
                self.current_image = None
                self.image_files = image_files
                self.current_image_index = 0
                
                pil_img = Image.open(self.image_files[0]).convert('RGB')
                self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                self.display_image(self.current_image)
                self.btn_detect.setEnabled(True)
                self.btn_prev.setEnabled(len(self.image_files) > 1)
                self.btn_next.setEnabled(len(self.image_files) > 1)
                self.status_bar.showMessage(f"已加载 {len(self.image_files)} 张图片，当前: {Path(self.image_files[0]).name}")
            else:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图像文件")
    
    def display_image(self, image: np.ndarray):
        if image is None:
            return
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        
        scaled_pixmap = pixmap.scaled(
            label_w, label_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def start_detection(self):
        if self.current_image is None:
            return
        
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        
        self.btn_detect.setEnabled(False)
        self.progress_bar.setValue(0)
        
        conf_threshold = self.conf_slider.value()
        iou_threshold = self.iou_slider.value()
        
        self.inference_thread = InferenceThread(self.model, self.current_image, self.image_size, conf_threshold, iou_threshold)
        self.inference_thread.progress.connect(self.progress_bar.setValue)
        self.inference_thread.finished.connect(self.on_detection_finished)
        self.inference_thread.start()
    
    def on_detection_finished(self, results, image):
        self.current_results = results
        
        result_image = image.copy()
        
        for r in results:
            bbox = r['bbox']
            conf = r['confidence']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Crack: {conf:.2f}"
            cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.display_image(result_image)
        
        self.update_results_text(results)
        
        self.btn_detect.setEnabled(True)
        self.status_bar.showMessage(f"Detection complete, found {len(results)} cracks")
    
    def update_results_text(self, results):
        text = f"Detection Results\n{'='*30}\n\n"
        text += f"Total detections: {len(results)}\n\n"
        
        if len(results) > 0:
            confs = [r['confidence'] for r in results]
            text += f"Confidence:\n"
            text += f"  Max: {max(confs):.4f}\n"
            text += f"  Min: {min(confs):.4f}\n"
            text += f"  Avg: {sum(confs)/len(confs):.4f}\n"
        
        self.results_text.setText(text)
    
    def export_result_image(self):
        if self.current_results is None or self.current_image is None:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", "",
            "JPEG (*.jpg);;PNG (*.png)"
        )
        
        if file_path:
            result_image = self.current_image.copy()
            for r in self.current_results:
                bbox = r['bbox']
                conf = r['confidence']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Crack: {conf:.2f}"
                cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(file_path, result_image)
            self.status_bar.showMessage(f"Result saved: {file_path}")
    
    def show_about(self):
        QMessageBox.about(
            self, "关于",
            "农田裂缝检测系统 v1.0\n\n"
            "基于YOLOv10目标检测\n"
            "用于农田裂缝检测\n\n"
            "作者: 高一峰\n"
            "指导教师: 田颖"
        )

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._load_current_image()

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self._load_current_image()

    def _load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            pil_img = Image.open(self.image_files[self.current_image_index]).convert('RGB')
            self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self.display_image(self.current_image)
            self.current_results = None

            from pathlib import Path
            img_name = Path(self.image_files[self.current_image_index]).name
            self.status_bar.showMessage(f"{len(self.image_files)} 张图片，当前: {img_name} ({self.current_image_index+1}/{len(self.image_files)})")

            self.btn_prev.setEnabled(self.current_image_index > 0)
            self.btn_next.setEnabled(self.current_image_index < len(self.image_files) - 1)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
