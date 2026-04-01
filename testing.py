from ultralytics import YOLO
import os

current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'saved_models', 'license_plate_best.pt')  # Original .pt
model = YOLO(model_path, task='detect')
model.export(format='openvino', simplify=True, dynamic=True, half=True)  # Dynamic batch + FP16