import os
import sys
import hashlib
import subprocess
import torch
import torchvision
import numpy as np

# Monkey-patch torchvision.ops.nms if C++ ops are missing
def custom_nms(boxes, scores, iou_threshold):
    import cv2
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    boxes_wh = boxes_np.copy()
    boxes_wh[:, 2] = boxes_np[:, 2] - boxes_np[:, 0]
    boxes_wh[:, 3] = boxes_np[:, 3] - boxes_np[:, 1]
    
    indices = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores_np.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    indices = np.array(indices).flatten()
    return torch.from_numpy(indices).to(device=boxes.device, dtype=torch.long)

if not hasattr(torchvision, 'ops'):
    class Ops: pass
    ops = Ops()
    ops.nms = custom_nms
    torchvision.ops = ops

from ultralytics import YOLO

def export_best_model():
    weights_path = "runs/detect/python_tools/runs/detect/yolo11_ball_game1/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "models/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolo11n.pt"
        print(f"Notice: Using base model '{weights_path}'")
        
    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)
    
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/yolov11_ttgames.onnx"
    engine_path = "models/yolov11_ttgames.engine"
    
    print(f"Exporting model to ONNX: {onnx_path}...")
    exported = model.export(
        format="onnx",
        dynamic=False,
        simplify=True,
        opset=12,
        imgsz=640
    )
    
    if os.path.exists(exported) and exported != onnx_path:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        os.rename(exported, onnx_path)
        
    print(f"ONNX export succeeded: {onnx_path}")
    
    # Compute SHA256 of ONNX model
    with open(onnx_path, "rb") as f:
        onnx_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"ONNX SHA-256: {onnx_hash}")
    
    # Compile TensorRT engine using trtexec if available
    trtexec_path = "trtexec"
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]
    print(f"Running TensorRT build command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode == 0 and os.path.exists(engine_path):
            print(f"TensorRT Engine build succeeded: {engine_path}")
            with open(engine_path, "rb") as f:
                engine_hash = hashlib.sha256(f.read()).hexdigest()
            print(f"Engine SHA-256: {engine_hash}")
            print(f"Engine Size: {os.path.getsize(engine_path)} bytes")
        else:
            print("trtexec stderr:", res.stderr)
            print("trtexec stdout:", res.stdout)
    except Exception as e:
        print(f"Could not build TensorRT engine via trtexec: {e}")

if __name__ == "__main__":
    export_best_model()
