import sys
from pathlib import Path
import json
import cv2
import numpy as np
import torch
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python_tools'))
from evaluate_blurball_candidate import build_network, import_file, preprocess
research = root / 'artifacts/blurball_research'
repo = research / 'source'
affine = import_file('diagnostic_affine', repo / 'src/utils/image.py')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = build_network(repo, research / 'blurball_best')
reports = []
with torch.inference_mode():
    for name in ['001', 'demo5']:
        directory = research / f'native_{name}_v2'
        native_frames = [cv2.imread(str(directory / f'source_first_{index}.png')) for index in [1,2,3]]
        capture = cv2.VideoCapture(str(Path('C:/Users/Adi/Downloads') / f'{name}.mp4'), cv2.CAP_FFMPEG)
        python_frames = [capture.read()[1] for _ in range(3)]
        capture.release()
        h,w = native_frames[0].shape[:2]
        transform = affine.get_affine_transform(np.array([w/2.,h/2.],np.float32), max(w,h),0,[512,288])
        native_input = np.fromfile(directory / 'input_first.bin', dtype=np.float32).reshape(1,9,288,512)
        reference_input = np.concatenate([preprocess(frame, transform) for frame in native_frames])[None]
        native_logits = np.fromfile(directory / 'native_logits.bin', dtype=np.float32).reshape(9,3,288,512)[0]
        identical_input_logits = model(torch.from_numpy(native_input).cuda())[0].float().cpu().numpy()[0]
        same_image_logits = model(torch.from_numpy(reference_input).cuda())[0].float().cpu().numpy()[0]
        reports.append({'clip':name, 'source_rgb_max_difference_per_frame':
                        [int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max()) for a,b in zip(native_frames,python_frames)],
                        'source_rgb_mean_difference_per_frame':
                        [float(np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()) for a,b in zip(native_frames,python_frames)],
                        'same_image_input_tensor_max_difference':float(np.abs(native_input-reference_input).max()),
                        'same_image_input_tensor_mean_difference':float(np.abs(native_input-reference_input).mean()),
                        'same_input_native_pytorch_max_logit_difference':float(np.abs(native_logits-identical_input_logits).max()),
                        'same_image_native_pytorch_max_logit_difference':float(np.abs(native_logits-same_image_logits).max()),
                        'python_opencv':cv2.__version__, 'native_opencv':'4.14.0'})
(research / 'native_preprocess_diagnostic.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2),flush=True)
