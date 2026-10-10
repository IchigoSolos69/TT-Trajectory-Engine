"""Research-only check of native BlurBall against the inspected upstream model."""
import sys
from pathlib import Path
import json
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python_tools'))
from evaluate_blurball_candidate import build_network, components, sha
from export_blurball_candidate import source_windows

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
research = root / 'artifacts/blurball_research'
repo = research / 'source'
model = build_network(repo, research / 'blurball_best')
records = []
max_logit = max_position = max_peak = max_extent = max_mass_relative = 0.
unmatched = 0
with torch.inference_mode():
    for directory in [research / 'native_001_v3', research / 'native_demo5_v3']:
        manifest = json.loads((directory / 'manifest.json').read_text())
        rows = [json.loads(line) for line in (directory / 'windows.jsonl').read_text().splitlines()]
        native = np.fromfile(directory / 'native_logits.bin', dtype=np.float32).reshape(len(rows), 3, 288, 512)
        inputs = np.fromfile(directory / 'native_inputs.bin', dtype=np.float32).reshape(len(rows), 1, 9, 288, 512)
        windows = source_windows(Path(manifest['source_path']), repo)
        if len(windows) != len(rows):
            raise RuntimeError('Mismatched window count')
        for index, (window, row) in enumerate(zip(windows, rows)):
            if row['source_frame_ids'] != window['source_frame_ids']:
                raise RuntimeError('Source frame mismatch')
            reference = model(torch.from_numpy(inputs[index]).cuda())[0].float().cpu().numpy()[0]
            delta = float(np.abs(reference - native[index]).max())
            max_logit = max(max_logit, delta)
            probabilities = torch.from_numpy(reference).sigmoid().numpy()
            record = {'source': manifest['source_path'], 'frame_ids': row['source_frame_ids'], 'max_logit_error': delta,
                      'channels': []}
            width, height = window['source_size']
            for channel in range(3):
                expected = components(probabilities[channel], window['inverse'], .7, width, height)
                actual = list(row['channels'][channel])
                unmatched += abs(len(expected) - len(actual))
                record['channels'].append({'pytorch_count': len(expected), 'native_count': len(actual)})
                for target in expected:
                    if not actual:
                        break
                    distances = [np.linalg.norm(np.array(target['center_px']) - value['center_px']) for value in actual]
                    match = int(np.argmin(distances)); candidate = actual.pop(match)
                    max_position = max(max_position, float(distances[match]))
                    max_peak = max(max_peak, abs(target['heatmap_peak'] - candidate['heatmap_peak']))
                    max_extent = max(max_extent, abs(target['blur_extent_source_px'] - candidate['blur_extent_source_px']))
                    max_mass_relative = max(max_mass_relative, abs(target['component_mass'] - candidate['component_mass']) /
                                            max(target['component_mass'], 1.e-9))
            records.append(record)
report = {'scope': 'TensorRT logits, weighted centroid and blur component parity with the identical native input tensors on 18 real source windows; OpenCV uint8 affine interpolation version difference documented separately; not accuracy or live latency.',
          'windows': records, 'max_logit_error': max_logit, 'max_center_delta_source_px': max_position,
          'max_peak_delta': max_peak, 'max_blur_extent_delta_source_px': max_extent,
          'max_component_mass_relative_delta': max_mass_relative, 'unmatched_components': unmatched,
          'engine_sha256': sha(research / 'export/blurball_fp32.engine'),
          'versions': {'torch': torch.__version__},
          'passed': max_logit < .002 and max_position < .01 and max_peak < .001 and max_extent < .01 and unmatched == 0}
(research / 'native_identical_input_parity.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'windows'}, indent=2), flush=True)
if not report['passed']:
    raise SystemExit('Native BlurBall parity failed')
