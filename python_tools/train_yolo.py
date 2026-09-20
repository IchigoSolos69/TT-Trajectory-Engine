import os
from ultralytics import YOLO

def train_custom_model():
    print("Initializing YOLOv11 PyTorch fine-tuning pipeline...")

    yaml_path = "dataset/data.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: Dataset configuration file not found at {yaml_path}")
        return

    # Load base model
    model = YOLO("yolo11n.pt")

    # Fine-tune the model for high-speed ball detection
    # Using device="cpu" and amp=False for max portability across PyTorch build architectures
    model.train(
        data=yaml_path,
        epochs=50,          # 50 epochs baseline
        imgsz=640,          # Matches C++ TensorRT buffer
        batch=8,            # CPU-optimized batch size
        device="cpu",       # CPU execution mode
        amp=False,          # Disable mixed precision check for CPU compatibility
        project="tt_models",
        name="tt_ball_tracker",
        exist_ok=True
    )

if __name__ == "__main__":
    train_custom_model()
