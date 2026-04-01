import cv2
import numpy as np
from ultralytics import YOLO
import re
from collections import defaultdict, deque, Counter
import difflib
import os
from tqdm import tqdm
import utility

# Get the current working directory
current_dir = os.getcwd()

# Dynamic paths
model_path = os.path.join(current_dir, 'saved_models', 'license_plate_best_openvino_model')
input_video_path = os.path.join(current_dir, 'cars_numberplate_02.mp4')
output_video_path = os.path.join(current_dir, 'output_video.mp4')

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = YOLO(model_path, task='detect')
model.tracker = 'botsort.yml'

# Regex pattern for typical license plate formats (adjust as needed)
plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

# Numberplate stabilization buffer
plate_history = defaultdict(lambda: deque(maxlen=10)) 
plate_final = {} 

def get_box_id(result):
    # Use YOLO tracking ID if available, else fallback to coordinates
    if result.id is not None:
        return str(int(result.id))  # Convert to string for consistency
    x1, y1, x2, y2 = map(int, result.xyxy[0])
    return f"{round(x1)}_{round(y1)}_{round(x2)}_{round(y2)}"

# Check if input video exists
if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"Input video not found at: {input_video_path}")

cap = cv2.VideoCapture(input_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
start_time = 5
end_time = 10

# Convert to Frame Numbers
start_frame = int(start_time*fps)
end_frame = int(end_time*fps)
num_frames_to_process = end_frame - start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # jump to start frame

# Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
CONF_THRESH = 0.5


with tqdm(total=num_frames_to_process, desc="Processing video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break

        # current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # print(f"\n--- Processing frame {current_frame_num} ---")

        # YOLO model on CPU
        results = model.track(frame, persist = True, verbose=False, device='intel:gpu')[0]
        
        for result in results.boxes:
            if result.conf[0] < CONF_THRESH:
                continue
            
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            box_id = get_box_id(result)
            print(f"Detection: box_id='{box_id}', bbox=({x1},{y1},{x2},{y2}), conf={result.conf[0]:.2f}")
            
            plate_crop = frame[y1:y2, x1:x2]
            
            # OCR using fast-plate-ocr and pattern matching
            plate_text = utility.recognize_plate(plate_crop)
            
            # Stabilization (updates plate_final internally)
            stable_plate = utility.get_stable_plate(box_id, plate_text, plate_history, plate_final, plate_pattern)
            
            # Removed redundant: if stable_plate: plate_final[box_id] = stable_plate
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Overlay zoomed-in plate above detected plate
            if plate_crop.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                overlay_drawn = False
                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized
                    overlay_drawn = True
                    # print(f"Overlay drawn at ({ox1},{oy1}) to ({ox2},{oy2})")

            # Show stabilized OCR text above overlay (or above bbox if overlay clipped)
            display_text = plate_final.get(box_id, "Reading...")
            # print(f"Display text for {box_id}: '{display_text}' (plate_final keys: {list(plate_final.keys())})")
            
            if display_text != "Reading...":  # Only show if stabilized
                # Use overlay pos if drawn, else bbox top; ensure y >= 20 to avoid off-screen
                text_y = oy1 - 20 if overlay_drawn else y1 - 20
                text_y = max(20, text_y)  # Clamp to visible area
                text_x = ox1 if overlay_drawn else x1
                
                # print(f"Drawing text '{display_text}' at ({text_x}, {text_y})")
                
                cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
        out.write(frame)
        pbar.update(1)

cap.release()
out.release()
# cv2.destroyAllWindows() 
print("Processing complete")