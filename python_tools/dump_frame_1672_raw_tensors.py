import json
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import cv2
import numpy as np
import onnxruntime as ort

def dump_frame_1672_onnx():
    model_path = "models/yolov11_ttgames.onnx"
    video_path = str(resolve_public_dataset_video("test_1"))

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1671) # 0-indexed position for 1672nd frame (1-based index 1672)
    ret, frame = cap.read()
    cap.release()

    imgW = 1920.0
    imgH = 1080.0

    # Aspect-Ratio Letterbox (640x640, padH=140)
    scaleR = min(640.0 / imgW, 640.0 / imgH) # 1/3
    unpadW = int(round(imgW * scaleR)) # 640
    unpadH = int(round(imgH * scaleR)) # 360
    padW = (640 - unpadW) // 2 # 0
    padH = (640 - unpadH) // 2 # 140

    resized = cv2.resize(frame, (unpadW, unpadH))
    padded = cv2.copyMakeBorder(resized, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

    outputs = session.run(None, {input_name: tensor})
    out = outputs[0][0].T # (8400, 5)

    # Sort by confidence descending
    sorted_indices = np.argsort(out[:, 4])[::-1]

    print("=========================================================================")
    print("  ONNX RAW TOP-5 ANCHORS FOR FRAME 1672 (Letterboxed Input)")
    print("=========================================================================")
    print("Rank | AnchorIdx | Raw 640x640 (cx, cy, w, h)     | Conf   | Unletterboxed 1920x1080 (X, Y) | Err to GT (797, 504)")
    print("---------------------------------------------------------------------------------------------------------------")

    gt_x, gt_y = 797.0, 504.0
    for rank in range(5):
        idx = sorted_indices[rank]
        cx, cy, w, h, conf = out[idx]
        pxX = (cx - padW) / scaleR
        pxY = (cy - padH) / scaleR
        err = np.sqrt((pxX - gt_x)**2 + (pxY - gt_y)**2)
        print(f" {rank+1:2d}  | {idx:9d} | ({cx:6.1f}, {cy:6.1f}, {w:5.1f}, {h:5.1f})     | {conf:6.3f} | ({pxX:6.1f}, {pxY:6.1f})                 | {err:6.2f}px")

if __name__ == "__main__":
    dump_frame_1672_onnx()
