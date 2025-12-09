"""
Simplified OOD Detection GUI - Working Version
"""
import sys
import os

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import cv2
import numpy as np
from pathlib import Path

# Add detection to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'detection'))
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                              QTextEdit, QScrollArea, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from detectron2.config import get_cfg


class InferenceWorker(QThread):
    finished = pyqtSignal(dict, np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, predictor, image_path, ood_threshold):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.ood_threshold = ood_threshold
        
    def run(self):
        try:
            self.progress.emit("Loading image...")
            image = cv2.imread(self.image_path)
            if image is None:
                self.error.emit(f"Failed to load: {self.image_path}")
                return
            
            self.progress.emit("Running detection...")
            height, width = image.shape[:2]
            
            # Prepare input
            input_im = [{
                "image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
                "height": height,
                "width": width
            }]
            
            # Run inference
            with torch.no_grad():
                instances = self.predictor(input_im)
            
            self.progress.emit("Processing results...")
            
            # NO filtering - use exact same settings as validation (SCORE_THRESH_TEST: 0.05)
            # The predictor already applies the 0.05 threshold internally
            
            # Extract results - instances is already the result, not a dict
            results = {
                'instances': instances,
                'logistic_score': instances.logistic_score if hasattr(instances, 'logistic_score') else None,
                'original_image': image
            }
            
            # Visualize
            vis_image = self.visualize(image, results, self.ood_threshold)
            
            self.finished.emit(results, vis_image)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def visualize(self, image, results, threshold):
        vis = image.copy()
        instances = results['instances']
        logistic_scores = results['logistic_score']
        
        if len(instances) == 0:
            return vis
        
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        # Use logistic scores if available
        if logistic_scores is not None:
            ood_scores = logistic_scores.cpu().numpy()
        else:
            ood_scores = np.ones(len(instances)) * 0.5
        
        class_names = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                       'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train',
                       'bottle', 'chair', 'dining table', 'potted plant', 'couch', 'tv']
        
        for box, cls, score, ood_score in zip(boxes, classes, scores, ood_scores):
            x1, y1, x2, y2 = box.astype(int)
            is_ood = ood_score < threshold
            # Red for OOD, Blue for ID (BGR format)
            color = (0, 0, 255) if is_ood else (255, 0, 0)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Simplified annotation: OOD shows only "OOD", ID shows class and confidence
            if is_ood:
                label = "OOD"
            else:
                class_name = class_names[cls] if cls < len(class_names) else f"cls_{cls}"
                label = f"{class_name} {score:.2f}"
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(vis, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        return vis


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.image_path = None
        self.worker = None
        
        self.initUI()
        self.loadModel()
        
    def initUI(self):
        self.setWindowTitle('VOS OOD Detection - Simplified')
        self.setGeometry(100, 100, 1400, 900)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Top button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.upload_btn = QPushButton('Select Image')
        self.upload_btn.clicked.connect(self.uploadImage)
        self.upload_btn.setEnabled(False)
        self.upload_btn.setMinimumHeight(50)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setFocusPolicy(Qt.StrongFocus)
        self.upload_btn.setStyleSheet("QPushButton { font-size: 14pt; font-weight: bold; padding: 10px 20px; background-color: #2196F3; color: white; border-radius: 5px; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #BDBDBD; }")
        button_layout.addWidget(self.upload_btn)
        
        self.detect_btn = QPushButton('Detect OOD')
        self.detect_btn.clicked.connect(self.runDetection)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setMinimumHeight(50)
        self.detect_btn.setCursor(Qt.PointingHandCursor)
        self.detect_btn.setFocusPolicy(Qt.StrongFocus)
        self.detect_btn.setStyleSheet("QPushButton { font-size: 14pt; font-weight: bold; padding: 10px 20px; background-color: #4CAF50; color: white; border-radius: 5px; } QPushButton:hover { background-color: #388E3C; } QPushButton:disabled { background-color: #BDBDBD; }")
        button_layout.addWidget(self.detect_btn)
        
        self.save_btn = QPushButton('Save Annotations')
        self.save_btn.clicked.connect(self.saveAnnotations)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(50)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setFocusPolicy(Qt.StrongFocus)
        self.save_btn.setStyleSheet("QPushButton { font-size: 14pt; font-weight: bold; padding: 10px 20px; background-color: #FF9800; color: white; border-radius: 5px; } QPushButton:hover { background-color: #F57C00; } QPushButton:disabled { background-color: #BDBDBD; }")
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        main_layout.addSpacing(10)
        
        # Middle row - images side by side
        image_layout = QHBoxLayout()
        
        # Original image
        orig_container = QVBoxLayout()
        orig_title = QLabel("Raw Image")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title_font = QFont()
        orig_title_font.setBold(True)
        orig_title_font.setPointSize(14)
        orig_title.setFont(orig_title_font)
        orig_container.addWidget(orig_title)
        
        self.orig_label = QLabel("No image loaded")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setScaledContents(False)
        self.orig_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orig_label.setMinimumSize(400, 300)
        self.orig_label.setStyleSheet("background: #f0f0f0; border: 2px solid #ccc; font-size: 12pt;")
        orig_container.addWidget(self.orig_label, 1)
        image_layout.addLayout(orig_container)
        
        # Annotated image
        result_container = QVBoxLayout()
        result_title = QLabel("Annotated Image")
        result_title.setAlignment(Qt.AlignCenter)
        result_title_font = QFont()
        result_title_font.setBold(True)
        result_title_font.setPointSize(14)
        result_title.setFont(result_title_font)
        result_container.addWidget(result_title)
        
        self.result_label = QLabel("Run detection first")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setScaledContents(False)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_label.setMinimumSize(400, 300)
        self.result_label.setStyleSheet("background: #f0f0f0; border: 2px solid #ccc; font-size: 12pt;")
        result_container.addWidget(self.result_label, 1)
        image_layout.addLayout(result_container)
        
        main_layout.addLayout(image_layout)
        main_layout.addSpacing(10)
        
        # Bottom - scrollable log
        log_title = QLabel("Detection Log")
        log_title_font = QFont()
        log_title_font.setBold(True)
        log_title_font.setPointSize(13)
        log_title.setFont(log_title_font)
        main_layout.addWidget(log_title)
        
        # Create scroll area for log
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(180)
        scroll_area.setMaximumHeight(250)
        
        self.detection_log = QTextEdit()
        self.detection_log.setReadOnly(True)
        self.detection_log.setFontFamily("Consolas")
        self.detection_log.setFontPointSize(11)
        self.detection_log.setLineWrapMode(QTextEdit.WidgetWidth)
        self.detection_log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.detection_log.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_area.setWidget(self.detection_log)
        main_layout.addWidget(scroll_area)
        
        self.log("Ready to detect OOD objects | Confidence: 0.30 | OOD Threshold: 0.8259")
        
    def log(self, msg):
        self.detection_log.append(msg)
    
    def resizeEvent(self, event):
        """Handle window resize to scale images"""
        super().resizeEvent(event)
        # Rescale original image if it exists
        if hasattr(self, 'image_path') and self.image_path and hasattr(self, '_orig_pixmap'):
            scaled = self._orig_pixmap.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.orig_label.setPixmap(scaled)
        # Rescale result image if it exists
        if hasattr(self, '_result_pixmap'):
            scaled = self._result_pixmap.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_label.setPixmap(scaled)
        
    def updateConfidence(self):
        # Method removed - no longer using adjustable confidence threshold
        pass
        
    def loadModel(self):
        try:
            self.log("Loading model...")
            
            cfg = get_cfg()
            from detection.core.setup import add_probabilistic_config
            add_probabilistic_config(cfg)
            cfg.merge_from_file("detection/configs/VOC-Detection/faster-rcnn/vos.yaml")
            cfg.merge_from_file("detection/configs/Inference/standard_nms.yaml")
            cfg.OUTPUT_DIR = "data/VOC-Detection/faster-rcnn/vos/random_seed_0"
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Set optimal confidence threshold for deployment (filters low-quality clutter)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.30  # Optimal: filters 30-45% clutter, keeps 85-90% main objects
            
            cfg.freeze()
            
            # Import here to avoid circular import
            from inference.inference_utils import build_predictor
            self.predictor = build_predictor(cfg)
            
            has_vos = (hasattr(self.predictor.model.roi_heads, 'logistic_regression') and 
                      hasattr(self.predictor.model.roi_heads, 'weight_energy'))
            
            info = f"Model: Faster R-CNN ResNet-50\n"
            info += f"VOS: {'Enabled' if has_vos else 'Disabled'}\n"
            info += f"Device: {cfg.MODEL.DEVICE}\n"
            info += f"Classes: 20 VOC\n\n"
            info += f"Optimal Threshold (Youden's Index):\n"
            info += f"  Confidence thresh: 0.30 (filters clutter)\n"
            info += f"  NMS thresh: 0.5\n"
            info += f"  OOD thresh: 0.8259\n"
            info += f"  AUROC: 98.79%, TPR: 94.74%, FPR: 3.43%"
            self.log("✓ Model loaded successfully")
            self.log(f"Device: {cfg.MODEL.DEVICE} | VOS: {'Enabled' if has_vos else 'Disabled'}")
            
            self.upload_btn.setEnabled(True)
            
        except Exception as e:
            self.log(f"[ERROR] {e}")
            import traceback
            self.log(traceback.format_exc())
            
    def uploadImage(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if path:
            self.image_path = path
            self.log(f"Loaded: {Path(path).name}")
            
            self._orig_pixmap = QPixmap(path)
            scaled = self._orig_pixmap.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.orig_label.setPixmap(scaled)
            
            self.detect_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.result_label.setText("Run detection")
            if hasattr(self, '_result_pixmap'):
                delattr(self, '_result_pixmap')
            
    def runDetection(self):
        if not self.image_path or not self.predictor:
            return
        
        self.log("\n" + "="*60)
        self.log("Running OOD Detection...")
        
        self.detect_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        
        # Use Youden's optimal threshold (98.79% AUROC, TPR=94.74%, FPR=3.43%)
        ood_threshold = 0.8259
        
        self.worker = InferenceWorker(self.predictor, self.image_path, ood_threshold)
        self.worker.finished.connect(self.onFinished)
        self.worker.error.connect(self.onError)
        self.worker.progress.connect(self.log)
        self.worker.start()
        
    def onFinished(self, results, vis_image):
        # Display
        h, w, c = vis_image.shape
        q_img = QImage(vis_image.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self._result_pixmap = QPixmap.fromImage(q_img)
        scaled = self._result_pixmap.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_label.setPixmap(scaled)
        
        # Stats
        instances = results['instances']
        logistic_scores = results['logistic_score']
        
        if len(instances) > 0:
            ood_threshold = 0.8259  # Youden's optimal threshold
            
            if logistic_scores is not None:
                ood_scores = logistic_scores.cpu().numpy()
                num_ood = (ood_scores < ood_threshold).sum()
                num_id = len(instances) - num_ood
                
                self.log(f"✓ Detection complete: {len(instances)} objects found")
                self.log(f"  → {num_id} In-Distribution (ID)")
                self.log(f"  → {num_ood} Out-of-Distribution (OOD)")
                self.log("="*60 + "\n")
                
                # Create detailed detection log
                self.createDetectionLog(instances, ood_scores, ood_threshold)
                self.save_btn.setEnabled(True)
            else:
                self.log(f"✓ Detection complete: {len(instances)} objects found")
                self.log("⚠ VOS scores not available")
                self.detection_log.clear()
                self.save_btn.setEnabled(True)
        else:
            self.log("✓ Detection complete: No objects detected")
            self.log("="*60)
            self.detection_log.clear()
            self.save_btn.setEnabled(False)
        
        self.detect_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
    
    def createDetectionLog(self, instances, ood_scores, ood_threshold):
        """Create detailed log of all detections"""
        class_names = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                       'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train',
                       'bottle', 'chair', 'dining table', 'potted plant', 'couch', 'tv']
        
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        log_text = "DETECTIONS:\n\n"
        
        for cls, conf, ood_score in zip(classes, scores, ood_scores):
            class_name = class_names[cls] if cls < len(class_names) else f"cls_{cls}"
            is_ood = ood_score < ood_threshold
            status = "OOD" if is_ood else "ID"
            
            log_text += f"{class_name} | {conf:.2f} | {status}\n"
        
        self.detection_log.setText(log_text)
    
    def saveAnnotations(self):
        """Save the annotated image to a selected folder"""
        if not hasattr(self, '_result_pixmap') or self._result_pixmap is None:
            self.log("✗ No annotated image to save")
            return
        
        # Get directory from user
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Annotations", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            # Create filename based on original image name
            original_name = Path(self.image_path).stem
            save_path = Path(folder) / f"{original_name}_annotated.png"
            
            # Save the pixmap
            if self._result_pixmap.save(str(save_path), "PNG"):
                self.log(f"✓ Annotations saved: {save_path.name}")
            else:
                self.log(f"✗ Failed to save annotations")
        
    def onError(self, msg):
        self.log(f"✗ Error: {msg}")
        self.log("="*60)
        self.detection_log.clear()
        self.detect_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
