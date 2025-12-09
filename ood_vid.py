"""
OOD Video Detection Script
Processes MP4 videos frame-by-frame with real-time OOD detection
Applies the same logic as ood_gui.py to video frames
"""
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import cv2
import numpy as np

# PyQt5 imports for GUI mode
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Add detection to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'detection'))
sys.path.insert(0, os.path.dirname(__file__))

from detectron2.config import get_cfg


class OODVideoProcessor:
    """
    Processes video files frame-by-frame with OOD detection.
    Uses the exact same logic as ood_gui.py for consistency.
    """
    
    def __init__(self, checkpoint_path=None, confidence_threshold=0.30, ood_threshold=0.8259):
        """
        Initialize the OOD video processor.
        
        Args:
            checkpoint_path (str): Path to model checkpoint (optional, uses default if None)
            confidence_threshold (float): Minimum confidence for detections (default: 0.30)
            ood_threshold (float): Threshold for OOD classification (default: 0.8259 - Youden's optimal)
        """
        self.predictor = None
        self.confidence_threshold = confidence_threshold
        self.ood_threshold = ood_threshold
        self.checkpoint_path = checkpoint_path
        
        # VOC class names (same as GUI)
        self.class_names = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train',
            'bottle', 'chair', 'dining table', 'potted plant', 'couch', 'tv'
        ]
        
        print("="*80)
        print("OOD Video Detection System")
        print("="*80)
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"OOD Threshold: {self.ood_threshold}")
        print("="*80)
        
    def load_model(self):
        """Load the VOS detection model (same logic as GUI)"""
        try:
            print("\n[1/4] Loading model configuration...")
            
            cfg = get_cfg()
            from detection.core.setup import add_probabilistic_config
            add_probabilistic_config(cfg)
            cfg.merge_from_file("detection/configs/VOC-Detection/faster-rcnn/vos.yaml")
            cfg.merge_from_file("detection/configs/Inference/standard_nms.yaml")
            cfg.OUTPUT_DIR = "data/VOC-Detection/faster-rcnn/vos/random_seed_0"
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Set optimal confidence threshold for deployment
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            
            # Override checkpoint if provided
            if self.checkpoint_path:
                cfg.MODEL.WEIGHTS = self.checkpoint_path
            
            cfg.freeze()
            
            print(f"[2/4] Building predictor on device: {cfg.MODEL.DEVICE}...")
            
            # Import here to avoid circular import
            from inference.inference_utils import build_predictor
            self.predictor = build_predictor(cfg)
            
            # Check if VOS is enabled
            has_vos = (hasattr(self.predictor.model.roi_heads, 'logistic_regression') and 
                      hasattr(self.predictor.model.roi_heads, 'weight_energy'))
            
            print(f"[3/4] Model loaded successfully!")
            print(f"      Model: Faster R-CNN ResNet-50")
            print(f"      VOS: {'Enabled ✓' if has_vos else 'Disabled ✗'}")
            print(f"      Device: {cfg.MODEL.DEVICE}")
            print(f"      Classes: 20 VOC classes")
            
            if not has_vos:
                print("\n⚠ WARNING: VOS not enabled! OOD detection may not work properly.")
                
            print("[4/4] Model ready for inference\n")
            
        except Exception as e:
            print(f"\n✗ ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def detect_frame(self, frame):
        """
        Run OOD detection on a single frame.
        Uses the same logic as InferenceWorker.run() in the GUI.
        
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV)
            
        Returns:
            dict: Detection results including instances and logistic scores
        """
        height, width = frame.shape[:2]
        
        # Prepare input (same as GUI)
        input_im = [{
            "image": torch.as_tensor(frame.transpose(2, 0, 1).astype("float32")),
            "height": height,
            "width": width
        }]
        
        # Run inference
        with torch.no_grad():
            instances = self.predictor(input_im)
        
        # Extract results
        results = {
            'instances': instances,
            'logistic_score': instances.logistic_score if hasattr(instances, 'logistic_score') else None,
            'original_frame': frame
        }
        
        return results
    
    def visualize_frame(self, frame, results):
        """
        Visualize detection results on a frame.
        Uses the exact same visualization logic as InferenceWorker.visualize() in the GUI.
        
        Args:
            frame (np.ndarray): Input frame
            results (dict): Detection results
            
        Returns:
            np.ndarray: Annotated frame
        """
        vis = frame.copy()
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
        
        for box, cls, score, ood_score in zip(boxes, classes, scores, ood_scores):
            x1, y1, x2, y2 = box.astype(int)
            is_ood = ood_score < self.ood_threshold
            
            # Red for OOD, Blue for ID (BGR format)
            color = (0, 0, 255) if is_ood else (255, 0, 0)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Simplified annotation: OOD shows only "OOD", ID shows class and confidence
            if is_ood:
                label = "OOD"
            else:
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"cls_{cls}"
                label = f"{class_name} {score:.2f}"
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(vis, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        return vis

    def draw_annotations(self, frame, boxes, classes, scores, ood_flags):
        """Draw boxes and labels onto a frame given arrays of detections."""
        vis = frame.copy()
        for box, cls, score, is_ood in zip(boxes, classes, scores, ood_flags):
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 0, 255) if is_ood else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if is_ood:
                label = "OOD"
            else:
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"cls_{cls}"
                label = f"{class_name} {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(vis, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        return vis

    @staticmethod
    def compute_iou(box, boxes):
        """Compute IoU between a box and an array of boxes. Box format: [x1,y1,x2,y2]"""
        if boxes is None or boxes.shape[0] == 0:
            return np.array([])
        x1 = np.maximum(box[0], boxes[:,0])
        y1 = np.maximum(box[1], boxes[:,1])
        x2 = np.minimum(box[2], boxes[:,2])
        y2 = np.minimum(box[3], boxes[:,3])
        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter = inter_w * inter_h
        area_box = max(0, (box[2]-box[0])) * max(0, (box[3]-box[1]))
        areas = np.maximum(0, boxes[:,2]-boxes[:,0]) * np.maximum(0, boxes[:,3]-boxes[:,1])
        union = area_box + areas - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou

    def extract_detection_arrays(self, results):
        """Return boxes, classes, scores, ood_flags as numpy arrays from results."""
        instances = results['instances']
        logistic_scores = results.get('logistic_score', None)
        if len(instances) == 0:
            return np.zeros((0,4)), np.zeros((0,), dtype=int), np.zeros((0,)), np.zeros((0,), dtype=bool)
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        if logistic_scores is not None:
            ood_scores = logistic_scores.cpu().numpy()
            ood_flags = ood_scores < self.ood_threshold
        else:
            ood_flags = np.zeros(len(boxes), dtype=bool)
        return boxes, classes, scores, ood_flags

    def draw_annotations(self, frame, boxes, classes, scores, ood_flags):
        """Draw boxes and labels onto a frame given arrays of detections."""
        vis = frame.copy()
        for box, cls, score, is_ood in zip(boxes, classes, scores, ood_flags):
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 0, 255) if is_ood else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if is_ood:
                label = "OOD"
            else:
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"cls_{cls}"
                label = f"{class_name} {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(vis, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        return vis

    @staticmethod
    def compute_iou(box, boxes):
        """Compute IoU between a box and an array of boxes. Box format: [x1,y1,x2,y2]"""
        if boxes is None or boxes.shape[0] == 0:
            return np.array([])
        x1 = np.maximum(box[0], boxes[:,0])
        y1 = np.maximum(box[1], boxes[:,1])
        x2 = np.minimum(box[2], boxes[:,2])
        y2 = np.minimum(box[3], boxes[:,3])
        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter = inter_w * inter_h
        area_box = max(0, (box[2]-box[0])) * max(0, (box[3]-box[1]))
        areas = np.maximum(0, boxes[:,2]-boxes[:,0]) * np.maximum(0, boxes[:,3]-boxes[:,1])
        union = area_box + areas - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou

    def extract_detection_arrays(self, results):
        """Return boxes, classes, scores, ood_flags as numpy arrays from results."""
        instances = results['instances']
        logistic_scores = results.get('logistic_score', None)
        if len(instances) == 0:
            return np.zeros((0,4)), np.zeros((0,), dtype=int), np.zeros((0,)), np.zeros((0,), dtype=bool)
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        if logistic_scores is not None:
            ood_scores = logistic_scores.cpu().numpy()
            ood_flags = ood_scores < self.ood_threshold
        else:
            ood_flags = np.zeros(len(boxes), dtype=bool)
        return boxes, classes, scores, ood_flags
    
    def get_detection_stats(self, results):
        """
        Extract detection statistics from results.
        
        Args:
            results (dict): Detection results
            
        Returns:
            dict: Statistics including total, ID, and OOD counts
        """
        instances = results['instances']
        logistic_scores = results['logistic_score']
        
        total = len(instances)
        num_id = 0
        num_ood = 0
        
        if total > 0 and logistic_scores is not None:
            ood_scores = logistic_scores.cpu().numpy()
            num_ood = int((ood_scores < self.ood_threshold).sum())
            num_id = total - num_ood
        
        return {
            'total': total,
            'id': num_id,
            'ood': num_ood
        }
    
    def process_video(self, input_video_path, output_video_path=None, save_frames=False, frames_dir=None, gui_mode=False, iou_threshold=0.30, id_iou=0.30):
        """
        Process a video file frame-by-frame with real-time OOD detection.
        
        Args:
            input_video_path (str): Path to input MP4 video
            output_video_path (str): Path to save output video (not used in GUI mode)
            save_frames (bool): Whether to save individual annotated frames
            frames_dir (str): Directory to save frames (if save_frames=True)
            gui_mode (bool): If True, display frames in GUI window instead of saving video
        """
        input_path = Path(input_video_path)
        
        if not input_path.exists():
            print(f"✗ ERROR: Input video not found: {input_path}")
            return
        
        # In GUI mode, output_path is optional
        if not gui_mode:
            output_path = Path(output_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_frames:
            if frames_dir is None:
                frames_dir = input_path.parent / f"{input_path.stem}_frames"
            else:
                frames_dir = Path(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            print(f"Frames will be saved to: {frames_dir}")
        
        print("="*80)
        print(f"Processing Video: {input_path.name}")
        if gui_mode:
            print("Mode: GUI Display (real-time visualization)")
        else:
            print(f"Mode: Save to file")
        print("="*80)
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"✗ ERROR: Could not open video: {input_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        print(f"Video Properties:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {timedelta(seconds=int(duration_sec))}")
        print("="*80)
        print(f"Configuration: Confidence={self.confidence_threshold}, OOD_Threshold={self.ood_threshold}, IoU_persistence={iou_threshold}, ID_IoU={id_iou}")
        print("="*80)
        
        # Initialize GUI window or video writer based on mode
        if gui_mode:
            # Create GUI application and window
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            window = VideoDisplayWindow(frame_width, frame_height)
            window.show()
            out = None
        else:
            # Create video writer for saving
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                print(f"✗ ERROR: Could not create output video writer")
                cap.release()
                return
            window = None
            app = None
        
        print("\nProcessing frames (respecting original video timing)...")
        print("="*80)
        
        # Statistics
        frame_count = 0
        processed_frame_count = 0
        skipped_frames = 0
        total_detections = 0
        total_id = 0
        total_ood = 0
        start_time = time.time()
        video_start_time = time.time()
        target_frame_time = 1.0 / fps if fps > 0 else 0.033
        
        # Temporal filtering: buffer previous frame's detections and only output them
        # if they persist into the next frame (IoU >= threshold). This reduces single-frame noise.
        # `iou_threshold` comes from the caller; `id_iou` controls the ID-locking/boosting IoU.

        # Read first frame to initialize buffer
        ret, prev_frame = cap.read()
        if not ret:
            print("✗ ERROR: Video has no frames")
            cap.release()
            if out is not None:
                out.release()
            return
        frame_count = 1
        prev_results = self.detect_frame(prev_frame)
        prev_boxes, prev_classes, prev_scores, prev_ood_flags = self.extract_detection_arrays(prev_results)
        processed_frame_count = 0
        write_index = 0

        while True:
            # GUI timing & skipping when GUI mode
            if gui_mode:
                if window is None or not window.isVisible():
                    print("\nGUI window closed by user. Stopping processing...")
                    break
                elapsed_real_time = time.time() - video_start_time
                expected_frame_index = int(elapsed_real_time * fps)
                if frame_count > expected_frame_index:
                    time.sleep(target_frame_time * 0.5)
                    continue
                frames_to_skip = max(0, expected_frame_index - frame_count)
                if frames_to_skip > 0:
                    for _ in range(frames_to_skip):
                        grabbed = cap.grab()
                        if not grabbed:
                            break
                        frame_count += 1
                        skipped_frames += 1

            # Read next frame (must decode to compare)
            ret, curr_frame = cap.read()
            if not ret:
                # No next frame -> drop prev boxes that don't persist (we have no next to match)
                # So do not output prev if it has any boxes (they would be single-frame). End loop.
                break

            frame_count += 1
            # Detect on current frame
            curr_results = self.detect_frame(curr_frame)
            curr_boxes, curr_classes, curr_scores, curr_ood_flags = self.extract_detection_arrays(curr_results)

            # --- ID Confidence Boosting ---
            # If a prev box was ID (not OOD) and a current box overlaps it (IoU >= iou_threshold)
            # then boost the current detection's confidence by doubling it (cap at 1.0).
            # This helps maintain tracking / persistence of previously-seen ID objects.
            try:
                if prev_boxes is not None and prev_boxes.shape[0] > 0 and curr_boxes is not None and curr_boxes.shape[0] > 0:
                    # Consider only previous in-distribution boxes
                    prev_id_mask = ~prev_ood_flags if prev_ood_flags is not None else np.ones(len(prev_boxes), dtype=bool)
                    prev_id_boxes = prev_boxes[prev_id_mask]
                    if prev_id_boxes.shape[0] > 0:
                        boosted = np.zeros(len(curr_scores), dtype=bool)
                        for pb in prev_id_boxes:
                            ious = self.compute_iou(pb, curr_boxes)
                            if ious.size == 0:
                                continue
                            # Find current indices with IoU >= ID locking threshold and which are ID
                            matches = np.where((ious >= id_iou) & (~curr_ood_flags))[0]
                            for mi in matches:
                                if not boosted[mi]:
                                    curr_scores[mi] = min(1.0, float(curr_scores[mi]) * 2.0)
                                    boosted[mi] = True
            except Exception:
                # Safety: do not let boosting crash the loop
                pass

            # For each prev box, check if any curr box has IoU >= threshold
            keep_mask = np.ones(len(prev_boxes), dtype=bool) if prev_boxes.shape[0] > 0 else np.array([], dtype=bool)
            for i in range(len(prev_boxes)):
                ious = self.compute_iou(prev_boxes[i], curr_boxes)
                if ious.size == 0 or np.max(ious) < iou_threshold:
                    # drop prev box as it didn't persist
                    keep_mask[i] = False

            kept_boxes = prev_boxes[keep_mask] if prev_boxes.shape[0] > 0 else prev_boxes
            kept_classes = prev_classes[keep_mask] if prev_classes.shape[0] > 0 else prev_classes
            kept_scores = prev_scores[keep_mask] if prev_scores.shape[0] > 0 else prev_scores
            kept_ood = prev_ood_flags[keep_mask] if prev_ood_flags.shape[0] > 0 else prev_ood_flags

            # Draw annotations on prev_frame using filtered detections
            annotated_prev = self.draw_annotations(prev_frame, kept_boxes, kept_classes, kept_scores, kept_ood)

            # Output annotated_prev: GUI display or write to video
            if gui_mode:
                window.update_frame(annotated_prev)
                app.processEvents()
            else:
                out.write(annotated_prev)

            write_index += 1
            if save_frames:
                frame_filename = frames_dir / f"frame_{write_index:06d}.jpg"
                cv2.imwrite(str(frame_filename), annotated_prev)

            # Update counters
            total_kept = kept_boxes.shape[0] if kept_boxes is not None else 0
            total_detections += int(total_kept)
            if total_kept > 0:
                total_ood += int(np.sum(kept_ood))
                total_id += int(total_kept - np.sum(kept_ood))
            processed_frame_count += 1

            # Shift buffer: current becomes previous
            prev_frame = curr_frame
            prev_boxes, prev_classes, prev_scores, prev_ood_flags = curr_boxes, curr_classes, curr_scores, curr_ood_flags

            # Logging
            if gui_mode:
                frame_time = time.time() - start_time
                video_time_remaining = (total_frames - frame_count) / fps if fps > 0 else 0
                eta_str = str(timedelta(seconds=int(video_time_remaining)))
                if processed_frame_count % 10 == 0 or frame_count == total_frames:
                    progress_pct = (frame_count / total_frames) * 100
                    skip_info = f" | Skipped: {skipped_frames}" if skipped_frames > 0 else ""
                    print(f"Frame {frame_count}/{total_frames} ({progress_pct:.1f}%) | Processed: {processed_frame_count} | "
                          f"Written: {total_kept} (ID: {int(total_kept - np.sum(kept_ood)) if total_kept>0 else 0}, OOD: {int(np.sum(kept_ood)) if total_kept>0 else 0}) | "
                          f"{skip_info} | ETA: {eta_str}")
            else:
                if processed_frame_count % 100 == 0 or frame_count == total_frames:
                    elapsed_time = time.time() - start_time
                    avg_time_per_frame = elapsed_time / processed_frame_count if processed_frame_count > 0 else 0
                    eta_seconds = (total_frames - frame_count) * avg_time_per_frame
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    progress_pct = (frame_count / total_frames) * 100
                    print(f"Frame {frame_count}/{total_frames} ({progress_pct:.1f}%) | Processed: {processed_frame_count} | "
                          f"Written last: {total_kept} (ID: {int(total_kept - np.sum(kept_ood)) if total_kept>0 else 0}, OOD: {int(np.sum(kept_ood)) if total_kept>0 else 0}) | "
                          f"AvgFrameTime: {avg_time_per_frame:.3f}s | ETA: {eta_str}")

        # End of buffered processing loop. We do not output the final prev_frame's boxes if they didn't persist (avoid single-frame noise).
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        if gui_mode and window is not None:
            window.close()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        avg_fps = processed_frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("="*80)
        print("Processing Complete!")
        print("="*80)
        print(f"Summary:")
        print(f"  Total Frames in Video: {total_frames}")
        print(f"  Frames Processed: {processed_frame_count}")
        print(f"  Frames Skipped: {skipped_frames}")
        print(f"  Total Time: {timedelta(seconds=int(elapsed_time))}")
        print(f"  Average Processing Speed: {avg_fps:.2f} FPS")
        print(f"  Video FPS: {fps:.2f}")
        print(f"  Total Detections: {total_detections}")
        print(f"    → In-Distribution (ID): {total_id}")
        print(f"    → Out-of-Distribution (OOD): {total_ood}")
        if not gui_mode:
            print(f"\nOutput saved to: {output_path}")
        if save_frames:
            print(f"Individual frames saved to: {frames_dir}")
        print("="*80)


class VideoDisplayWindow(QMainWindow):
    """
    GUI window for displaying video frames in real-time.
    """
    def __init__(self, width, height):
        super().__init__()
        self.setWindowTitle("OOD Video Detection - Real-time Display")
        
        # Create label to display frames
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.setCentralWidget(self.image_label)
        
        # Set window size
        self.resize(width, height)
        
    def update_frame(self, frame):
        """
        Update the displayed frame.
        
        Args:
            frame: OpenCV BGR frame (numpy array)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to window size while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Display
        self.image_label.setPixmap(scaled_pixmap)



def main():
    parser = argparse.ArgumentParser(
        description="Process MP4 videos with OOD detection frame-by-frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - save video
  python ood_video_detector.py input.mp4 output.mp4
  
  # GUI mode - display frames in real-time (no video saved)
  python ood_video_detector.py input.mp4 output.mp4 --gui
  
  # With custom thresholds
  python ood_video_detector.py input.mp4 output.mp4 --confidence 0.30 --ood-threshold 0.8259
  
  # Save individual frames
  python ood_video_detector.py input.mp4 output.mp4 --save-frames --frames-dir output_frames
  
  # Use custom checkpoint
  python ood_video_detector.py input.mp4 output.mp4 --checkpoint path/to/model.pth
        """
    )
    
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input MP4 video file'
    )
    
    parser.add_argument(
        'output_video',
        type=str,
        nargs='?',
        default=None,
        help="(Optional) Path to save output video with annotations. If omitted, will be input filename + '_annotated'"
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.30,
        help='Confidence threshold for detections (default: 0.30)'
    )
    
    parser.add_argument(
        '--ood-threshold',
        type=float,
        default=0.8259,
        help='OOD classification threshold (default: 0.8259 - Youden\'s optimal)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional, uses default if not specified)'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual annotated frames as images'
    )
    
    parser.add_argument(
        '--frames-dir',
        type=str,
        default=None,
        help='Directory to save individual frames (default: <output_video>_frames/)'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Display frames in real-time GUI window instead of saving video'
    )

    parser.add_argument(
        '--IoU',
        type=float,
        default=0.30,
        help='IoU persistence threshold for temporal filtering (default: 0.30)'
    )

    parser.add_argument(
        '--ID_IoU',
        type=float,
        default=0.30,
        help='IoU threshold for ID locking/boosting (default: 0.30)'
    )
    
    args = parser.parse_args()
    
    # Determine output path (auto-generate if not provided)
    input_path_arg = Path(args.input_video)
    if args.output_video is None:
        # Keep same extension as input; insert _annotated before suffix
        out_path = input_path_arg.with_name(input_path_arg.stem + "_annotated" + input_path_arg.suffix)
    else:
        out_path = Path(args.output_video)

    # Create processor
    processor = OODVideoProcessor(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.confidence,
        ood_threshold=args.ood_threshold
    )

    # Load model
    processor.load_model()

    # Process video
    processor.process_video(
        input_video_path=str(input_path_arg),
        output_video_path=str(out_path),
        save_frames=args.save_frames,
        frames_dir=args.frames_dir,
        gui_mode=args.gui,
        iou_threshold=args.IoU,
        id_iou=args.ID_IoU
    )
    
    print("\n✓ All done!")


if __name__ == '__main__':
    main()
