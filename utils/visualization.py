import cv2
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from collections import Counter

class ResultVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.line_thickness = 2
        
        self.class_names = {0: '轻度', 1: '中度', 2: '重度'}
        self.class_colors = {
            0: (0, 255, 0),
            1: (0, 255, 255),
            2: (0, 0, 255)
        }
    
    def draw_detections(self, 
                        image: np.ndarray, 
                        detections: List[Dict],
                        show_confidence: bool = True) -> np.ndarray:
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det.get('class_id', 0)
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.line_thickness)
            
            label = self.class_names.get(class_id, str(class_id))
            if show_confidence and 'score' in det:
                label += f" {det['score']:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, 1
            )
            
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                result_image, label,
                (x1, y1 - 5),
                self.font, self.font_scale,
                (255, 255, 255), 1
            )
        
        return result_image
    
    def draw_statistics_panel(self, 
                              image: np.ndarray,
                              detections: List[Dict]) -> np.ndarray:
        h, w = image.shape[:2]
        
        panel_width = 250
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        
        class_counts = Counter([self.class_names.get(d.get('class_id', 0), '未知') for d in detections])
        
        y_offset = 30
        cv2.putText(panel, "检测统计", (10, y_offset), 
                   self.font, 0.9, (255, 255, 255), 2)
        
        y_offset += 40
        cv2.putText(panel, f"总数: {len(detections)}", (10, y_offset),
                   self.font, 0.7, (255, 255, 255), 1)
        
        colors = {'轻度': (0, 255, 0), '中度': (0, 255, 255), '重度': (0, 0, 255)}
        
        for class_name, count in class_counts.items():
            y_offset += 30
            cv2.putText(panel, f"{class_name}: {count}", (10, y_offset),
                       self.font, 0.7, colors.get(class_name, (255, 255, 255)), 1)
        
        combined = np.hstack([image, panel])
        
        return combined
    
    def create_detection_report(self, 
                                detections: List[Dict],
                                output_path: str):
        class_counts = Counter([self.class_names.get(d.get('class_id', 0), '未知') for d in detections])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = ['#90EE90', '#FFD700', '#FF6347']
        
        axes[0].pie(sizes, labels=labels, colors=colors[:len(labels)],
                   autopct='%1.1f%%', startangle=90)
        axes[0].set_title('干裂程度分布')
        
        bars = axes[1].bar(labels, sizes, color=colors[:len(labels)])
        axes[1].set_xlabel('干裂等级')
        axes[1].set_ylabel('检测数量')
        axes[1].set_title('各等级检测数量')
        
        for bar, size in zip(bars, sizes):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def export_results(self, 
                       detections: List[Dict],
                       image_path: str,
                       output_format: str = 'json'):
        import json
        from datetime import datetime
        
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(detections),
            'detections': detections
        }
        
        if output_format == 'json':
            output_path = image_path.rsplit('.', 1)[0] + '_results.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
