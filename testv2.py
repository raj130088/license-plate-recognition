import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from ultralytics.engine.results import Results
from ultralytics.utils.ops import Boxes
from ultralytics.trackers.botsort import BOTSORT
from ultralytics.utils import yaml_load
import re
from collections import defaultdict, deque, Counter
import difflib
import os
from tqdm import tqdm
import utility
import time 
import openvino as ov
import openvino.properties.hint as hints
import torch
import torchvision

# Get the current working directory
current_dir = os.getcwd()

# Dynamic paths
model_path = os.path.join(current_dir, 'saved_models', 'license_plate_best_openvino_model')
input_video_path = os.path.join(current_dir, 'cars_numberplate_02.mp4')
output_video_path = os.path.join(current_dir, 'output_video.mp4')

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load OpenVINO model with throughput mode
core = ov.Core()
xml_path = os.path.join(model_path, "license_plate_best.xml")
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML model file not found at: {xml_path}")

model_ov = core.read_model(xml_path)
nc = model_ov.outputs[0].get_partial_shape()[1].get_max_length() - 4  # num_classes = output_dim - 4 (box)
config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
compiled_model = core.compile_model(model_ov, device_name="GPU", config=config)

# Load tracker
tracker_args = yaml_load('botsort.yml')
tracker = BOTSORT(args=tracker_args)

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

def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114), auto=False, scale_fill=False, scaleup=False, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# Check if input video exists
if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"Input video not found at: {input_video_path}")

cap = cv2.VideoCapture(input_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
start_time = 5
end_time = 10

# Convert to Frame Numbers
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)
num_frames_to_process = end_frame - start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # jump to start frame

# Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
CONF_THRESH = 0.5

# Batching setup
BATCH_SIZE = 4  # Start here; reduce to 2 if VRAM spikes in Task Manager
frame_buffer = []
processed_count = 0

start_time_total = time.time()  # Optional: For overall timing

# Helper function
def _process_batch(frames, compiled_model, tracker, out, plate_history, plate_final, plate_pattern, conf_thresh, nc):
    input_hw = (640, 640)
    processed = []
    ratios = []
    pads = []
    for frame in frames:
        im, ratio, (dw, dh) = letterbox(frame, new_shape=input_hw)
        im = im[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR to RGB, HWC to CHW, 0-1
        processed.append(im)
        ratios.append(ratio)
        pads.append((dw, dh))

    batched = np.stack(processed, 0).astype(np.float32)  # (B,3,H,W)

    outputs = compiled_model([batched])[0]  # (B,5,8400)

    # Post-process and track sequentially
    for i in range(len(frames)):
        pred = outputs[i].T  # (8400,5)
        pred = torch.from_numpy(pred)
        dets = non_max_suppression(pred.unsqueeze(0), conf_thres=conf_thresh, iou_thres=0.7, nc=nc, max_det=300)[0]

        if len(dets):
            dets[:, :4] = scale_boxes(input_hw, dets[:, :4], frames[i].shape[:2], ratios[i], pads[i], None)

        # Create Results for tracker
        if len(dets):
            dets = torch.cat((dets[:, :4], dets[:, 4:5], torch.zeros(len(dets), 1, dtype=dets.dtype)), 1)  # add cls=0
            boxes = Boxes(dets, orig_shape=frames[i].shape)
        else:
            boxes = Boxes(torch.empty(0, 6), orig_shape=frames[i].shape)
        results = Results(orig_img=frames[i], path='', names={0: 'license_plate'}, boxes=boxes)

        # Update tracker
        results = tracker.update(results)

        # Now process detections with tracking IDs
        for result in results.boxes:
            if result.conf[0] < conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, result.xyxy[0])
            box_id = get_box_id(result)
            print(f"Detection: box_id='{box_id}', bbox=({x1},{y1},{x2},{y2}), conf={result.conf[0]:.2f}")

            plate_crop = frames[i][y1:y2, x1:x2]

            # OCR using fast-plate-ocr and pattern matching
            plate_text = utility.recognize_plate(plate_crop)

            # Stabilization (updates plate_final internally)
            stable_plate = utility.get_stable_plate(box_id, plate_text, plate_history, plate_final, plate_pattern)

            # Draw bounding box
            cv2.rectangle(frames[i], (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Overlay zoomed-in plate above detected plate
            if plate_crop.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                overlay_drawn = False
                if oy2 <= frames[i].shape[0] and ox2 <= frames[i].shape[1]:
                    frames[i][oy1:oy2, ox1:ox2] = plate_resized
                    overlay_drawn = True
                    # print(f"Overlay drawn at ({ox1},{oy1}) to ({ox2},{oy2})")

            # Show stabilized OCR text above overlay (or above bbox if overlay clipped)
            display_text = plate_final.get(box_id, "Reading...")
            # print(f"Display text for {box_id}: '{display_text}' (plate_final.keys(): {list(plate_final.keys())})")
            
            if display_text != "Reading...":  # Only show if stabilized
                # Use overlay pos if drawn, else bbox top; ensure y >= 20 to avoid off-screen
                text_y = oy1 - 20 if overlay_drawn else y1 - 20
                text_y = max(20, text_y)  # Clamp to visible area
                text_x = ox1 if overlay_drawn else x1
                
                # print(f"Drawing text '{display_text}' at ({text_x}, {text_y})")
                
                cv2.putText(frames[i], display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                cv2.putText(frames[i], display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        out.write(frames[i])  # Write the processed frame

with tqdm(total=num_frames_to_process, desc="Processing video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            # Process any remaining frames
            if frame_buffer:
                _process_batch(frame_buffer, compiled_model, tracker, out, plate_history, plate_final, plate_pattern, CONF_THRESH, nc)
            break

        frame_buffer.append(frame)
        pbar.update(1)

        # When buffer is full, process the batch
        if len(frame_buffer) == BATCH_SIZE:
            _process_batch(frame_buffer, compiled_model, tracker, out, plate_history, plate_final, plate_pattern, CONF_THRESH, nc)
            frame_buffer.clear()
            processed_count += BATCH_SIZE

# Flush any leftovers (but safety)
if frame_buffer:
    _process_batch(frame_buffer, compiled_model, tracker, out, plate_history, plate_final, plate_pattern, CONF_THRESH, nc)

end_time_total = time.time()
print(f"Processing complete in {end_time_total - start_time_total:.2f}s ({processed_count / (end_time_total - start_time_total):.2f} fps)")

cap.release()
out.release()
print("Processing complete")