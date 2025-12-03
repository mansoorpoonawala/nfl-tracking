from ultralytics import YOLO
import torch

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Load pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train with multi-GPU support
results = model.train(
    data="data_combined.yaml",
    epochs=500,
    imgsz=640,
    batch=64,
    device=[0, 1, 2, 3],
    name="nfl_combined_500ep",
    project="runs",
    exist_ok=True,
    patience=0,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    amp=True,
    workers=8,
    save=True,
    save_period=10,
)

print("Training complete!")
print(f"Best results saved to: {results.save_dir}")
