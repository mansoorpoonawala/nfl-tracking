from ultralytics import YOLO
import torch
import os

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Load pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train with multi-GPU support - use only available GPUs on this node
available_gpus = list(range(torch.cuda.device_count()))

# Train with multi-GPU support
results = model.train(
    data="data_combined.yaml",
    epochs=500,
    imgsz=640,
    batch=128,
    device=available_gpus,  # Use only GPUs available on this node
    name="nfl_combined_500ep_8gpu",
    project="runs",
    exist_ok=True,
    patience=0,
    deterministic=True,
    plots=True,
    save_period=10,
    optimizer="AdamW",
)
