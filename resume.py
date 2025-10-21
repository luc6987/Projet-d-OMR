from ultralytics import YOLO

# 加载checkpoint
model = YOLO('runs/detect/yolov8l_muscima_finetune3/weights/last.pt')

# 继续训练
results = model.train(
    data='MyCoolDataset/data.yaml',
    epochs=500,
    batch=8,
    imgsz=640,
    lr0=5.5e-5,
    optimizer='AdamW',
    patience=100,
    project='runs/detect',
    name='yolov8l_muscima_resume',
    resume=True  # 这里使用True而不是路径
)