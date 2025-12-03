import os
import torch
from ultralytics import YOLO

# Set up distributed environment
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
local_rank = int(os.environ.get('LOCAL_RANK', 0))

print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Load model
model = YOLO("yolo11n.pt")

# Train - let PyTorch handle distributed training
results = model.train(
    data="data_combined.yaml",
    epochs=500,
    imgsz=640,
    batch=64,  # Per GPU batch size
    device=local_rank,  # Use only local GPU
    name="nfl_combined_500ep_distributed",
    project="runs",
    exist_ok=True,
    patience=0,
    deterministic=True,
    plots=True,
    save_period=10,
    optimizer="AdamW",
)
