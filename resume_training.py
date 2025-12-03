from ultralytics import YOLO

model = YOLO("runs/nfl_combined_500ep/weights/last.pt")

results = model.train(
    data="data_combined.yaml",
    epochs=500,
    imgsz=640,
    batch=64,
    device=[0,1,2,3],
    name="nfl_combined_500ep_resumed",
    project="runs",
    exist_ok=True,
    patience=0,
)
