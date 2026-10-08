"""Compare deployed TensorRT detector responses; scores are NOT accuracy labels."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import tensorrt as trt


class Detector:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if not self.engine:
            raise RuntimeError('Cannot load deployed TensorRT engine')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(n < 1 for n in shape):
                raise ValueError('Expected fixed-shape deployed engine')
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in (trt.float32, trt.float16):
                raise ValueError('Unsupported deployed tensor dtype')
            self.tensors[name] = torch.empty(shape, device='cuda', dtype=torch.float32 if dtype == trt.float32 else torch.float16)
            self.context.set_tensor_address(name, self.tensors[name].data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input = name
            else:
                self.output = name

    def infer(self, frame):
        candidates = self.infer_candidates(frame)
        return candidates[0]

    def infer_candidates(self, frame, limit=20):
        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        uw, uh = round(w * scale), round(h * scale)
        left, top = (640 - uw) // 2, (640 - uh) // 2
        resized = cv2.resize(frame, (uw, uh), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, top, 640 - uh - top, left, 640 - uw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
        with torch.cuda.stream(self.stream):
            self.tensors[self.input].copy_(torch.from_numpy(array).unsqueeze(0))
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')
        self.stream.synchronize()
        raw = self.tensors[self.output][0].float().cpu().numpy()
        if raw.shape != (5, 8400) or not np.isfinite(raw).all():
            raise RuntimeError('Unexpected/nonfinite detector output')
        ordered = np.argsort(raw[4])[::-1]
        candidates = []
        for i in ordered:
            x, y = (float(raw[0, i]) - left) / scale, (float(raw[1, i]) - top) / scale
            bw, bh = float(raw[2, i]) / scale, float(raw[3, i]) / scale
            score = float(raw[4, i])
            box = [x - bw / 2, y - bh / 2, bw, bh]
            # Class-0-only overlap suppression for diagnostic proposals.
            overlap = False
            for previous in candidates:
                px, py, pw, ph = previous['box_px']
                iw = max(0., min(box[0] + bw, px + pw) - max(box[0], px))
                ih = max(0., min(box[1] + bh, py + ph) - max(box[1], py))
                if iw * ih / max(1e-9, bw * bh + pw * ph - iw * ih) > .45:
                    overlap = True; break
            if overlap:
                continue
            candidates.append({'score': score, 'center_px': [x, y], 'box_px': box,
                               'accepted_at_0_25': score >= .25 and 0 <= x < w and 0 <= y < h})
            if len(candidates) == limit:
                break
        return candidates


def variants(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]
    median = float(np.median(luminance[::8, ::8]))
    gamma = max(1., min(1.35, np.log(.55) / np.log(max(median / 255., .55))))
    lut = np.rint((np.arange(256) / 255.) ** gamma * 255).astype(np.uint8)
    adjusted = lab.copy()
    adjusted[:, :, 0] = cv2.LUT(luminance, lut)
    clahe = lab.copy()
    clahe[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(luminance)
    adaptive = adjusted.copy()
    if median > 160:
        adaptive[:, :, 0] = cv2.addWeighted(adjusted[:, :, 0], .65, cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(adjusted[:, :, 0]), .35, 0)
    else:
        adaptive = lab.copy()
    return {'raw': frame, 'gamma': cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR),
            'clahe': cv2.cvtColor(clahe, cv2.COLOR_LAB2BGR),
            'adaptive': cv2.cvtColor(adaptive, cv2.COLOR_LAB2BGR) if median > 160 else frame}, {'median_luma': median, 'gamma': gamma, 'adaptive_applied': median > 160}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    detector = Detector(args.engine)
    capture = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError('Video unavailable')
    rows = []
    try:
        cursor = 0
        if not args.frames or any(fid < 1 for fid in args.frames):
            raise ValueError('Provide positive source-frame IDs')
        for fid in sorted(set(args.frames)):
            while cursor < fid:
                if not capture.grab():
                    raise RuntimeError(f'Cannot decode through source frame {fid}')
                cursor += 1
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {fid}')
            images, diagnostics = variants(frame)
            results = {name: detector.infer(image) for name, image in images.items()}
            rows.append({'frame_id': fid, 'source_pts_ms': capture.get(cv2.CAP_PROP_POS_MSEC), **diagnostics, 'variants': results})
            if fid == args.frames[0] or fid == 5202:
                panels = []
                for name, image in images.items():
                    panel = cv2.resize(image, (768, 432))
                    result = results[name]
                    x, y = result['center_px']
                    if result['accepted_at_0_25']:
                        cv2.circle(panel, (round(x * 768 / frame.shape[1]), round(y * 432 / frame.shape[0])), 8, (0, 180, 255), 2)
                    cv2.putText(panel, f"{name}: score {result['score']:.3f} (unreviewed)", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
                    panels.append(panel)
                if not cv2.imwrite(str(args.output / f'comparison_{fid}.jpg'), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])):
                    raise RuntimeError('Cannot save comparison image')
            print(f'frame {fid}: ' + ', '.join(f"{name}={result['score']:.3f}" for name, result in results.items()), flush=True)
    finally:
        capture.release()
    report = {'video': str(args.video), 'decode_policy': 'Sequential source-frame enumeration; no random seek', 'engine_sha256': hashlib.sha256(args.engine.read_bytes()).hexdigest(),
              'scope': 'Unlabeled offline TensorRT response comparison, not native C++ parity, precision/recall or live performance',
              'promoted': False, 'accuracy': None, 'accuracy_reason': 'No reviewed ground truth for these frames', 'frames': rows}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path); parser.add_argument('engine', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--frames', nargs='+', type=int, default=[1000, 2500, 5202, 5203, 5204, 5205, 9000, 15000, 20000])
    main(parser.parse_args())
