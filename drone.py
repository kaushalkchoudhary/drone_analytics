import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import supervision as sv

def process_video():
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    input_path = os.path.join(project_root, "public", "drone8.mp4")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "runs", "drone_analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"drone_speed_analysis_{timestamp}.mp4")

    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # Initialize VideoCapture
    cap = cv2.VideoCapture(input_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if video_fps == 0:
        print("Error: Could not read video properties.")
        return

    # Target 10 FPS for AI and Output
    target_fps = 10
    stride = max(1, int(round(video_fps / target_fps)))
    actual_output_fps = video_fps / stride

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, actual_output_fps, (width, height))

    print(f"--- Tactical Speed Analysis Mode ---")
    print(f"No-Calibration Mode | Output FPS: {actual_output_fps:.2f}")
    print(f"-------------------------------------\n")

    model = YOLO("yolov8s-visdronemodel.pt")
    heatmap_buffer = np.zeros((height, width), dtype=np.float32)
    
    # Trackers and Annotators
    tracker = sv.ByteTrack(frame_rate=actual_output_fps)
    box_annotator = sv.BoxAnnotator(thickness=2)
    # Using LabelAnnotator with custom labels for speed classes
    label_annotator = sv.LabelAnnotator(
        text_padding=4,
        text_scale=0.4,
        text_thickness=1,
        color_lookup=sv.ColorLookup.CLASS
    )

    # Speed Tracking State
    # dict mapping tracker_id to (last_cx, last_cy)
    previous_positions = {}
    # thresholds in pixels per second (approximate for drone height)
    THRESH_STALLED = 15
    
    THRESH_SLOW = 80
    THRESH_MEDIUM = 250

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            start_time = time.time()
            results = model.predict(
                frame, 
                conf=0.10, 
                verbose=False, 
                max_det=500,
                classes=[3, 4, 5, 8]
            )[0]
            
            detections = sv.Detections.from_ultralytics(results)
            current_detections = tracker.update_with_detections(detections)
            
            # --- SYNCHRONIZED HEATMAP & SPEED CALC ---
            heatmap_buffer *= 0.95
            speed_labels = []
            
            category_counts = {"STALLED": 0, "SLOW": 0, "MEDIUM": 0, "FAST": 0}
            
            if current_detections is not None and current_detections.tracker_id is not None:
                for i, (xyxy, tid) in enumerate(zip(current_detections.xyxy, current_detections.tracker_id)):
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    
                    # 1. Update Heatmap
                    if 0 <= cx < width and 0 <= cy < height:
                        cv2.circle(heatmap_buffer, (cx, cy), radius=10, color=2.0, thickness=-1)
                    
                    # 2. Calculate Speed (Relative Pixel Delta)
                    speed_val = 0
                    if tid in previous_positions:
                        prev_cx, prev_cy = previous_positions[tid]
                        dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                        # Convert to pixels per second
                        speed_val = dist * actual_output_fps
                    
                    previous_positions[tid] = (cx, cy)
                    
            # 3. Categorize and Assign Colors
                    if speed_val < THRESH_STALLED:
                        cat = "STALLED"
                        cid = 0 # Red index
                    elif speed_val < THRESH_SLOW:
                        cat = "SLOW"
                        cid = 1 # Orange index
                    elif speed_val < THRESH_MEDIUM:
                        cat = "MEDIUM"
                        cid = 2 # Yellow index
                    else:
                        cat = "FAST"
                        cid = 3 # Green index
                    
                    category_counts[cat] += 1
                    speed_labels.append(f"#{tid} {cat}")
                    current_detections.class_id[i] = cid

            # Cleanup old IDs from previous_positions to save memory
            active_ids = set(current_detections.tracker_id) if current_detections.tracker_id is not None else set()
            previous_positions = {tid: pos for tid, pos in previous_positions.items() if tid in active_ids}

            # --- CONGESTION CALCULATION ---
            total_objs = len(active_ids)
            if total_objs > 0:
                # Weighted formula: Stalled (1.0) + Slow (0.6) + Med (0.1)
                congestion_score = (category_counts['STALLED'] * 1.0) + (category_counts['SLOW'] * 0.6) + (category_counts['MEDIUM'] * 0.1)
                congestion_pct = min(100, int((congestion_score / total_objs) * 100))
            else:
                congestion_pct = 0

            # --- VISUALIZATION ---
            smoothed = cv2.GaussianBlur(heatmap_buffer, (31, 31), 0)
            local_max = np.max(smoothed)
            
            display_frame = frame.copy()
            if local_max > 0:
                norm_map = (smoothed / local_max * 255).astype(np.uint8)
                jet_overlay = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                mask = norm_map > 15
                display_frame[mask] = cv2.addWeighted(frame[mask], 0.55, jet_overlay[mask], 0.45, 0)

            # --- SPEED COLOR PALETTE ---
            # 0: STALLED (Red), 1: SLOW (Orange), 2: MEDIUM (Yellow), 3: FAST (Green)
            custom_palette = sv.ColorPalette(colors=[
                sv.Color(r=255, g=0, b=0),      # Stalled
                sv.Color(r=255, g=165, b=0),    # Slow
                sv.Color(r=255, g=255, b=0),    # Medium
                sv.Color(r=0, g=255, b=0)       # Fast
            ])
            box_annotator.color = custom_palette
            label_annotator.color = custom_palette

            # Annotate BBoxes
            if current_detections is not None:
                display_frame = box_annotator.annotate(scene=display_frame, detections=current_detections)
                # Only show labels if density is low (<= 50)
                if total_objs <= 50:
                    display_frame = label_annotator.annotate(scene=display_frame, detections=current_detections, labels=speed_labels)

            # --- HUD OVERLAY: CONGESTION INDEX ---
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, "CONGESTION INDEX", (w - 300, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            # Dynamic Color for %: Green -> Red
            col = (0, 255 - int(2.5 * congestion_pct), int(2.5 * congestion_pct))
            cv2.putText(display_frame, f"{congestion_pct}%", (w - 200, 130), cv2.FONT_HERSHEY_DUPLEX, 2.0, col, 3)

            # SAVE FRAME
            out.write(display_frame)
            
            # Real-time Reporting
            print(f"Frm {frame_idx:<5} | Objs: {total_objs:<3} | Index: {congestion_pct}% | {category_counts['STALLED']} Stalled | {category_counts['SLOW']} Slow | {category_counts['MEDIUM']} Med")

        frame_idx += 1
        if frame_idx % 300 == 0:
            sys.stdout.flush()

    cap.release()
    out.release()
    print("-" * 50)
    print(f"Speed Analysis Video saved: {output_path}")

if __name__ == "__main__":
    process_video()

