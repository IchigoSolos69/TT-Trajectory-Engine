"""Read saved labels and dependency metadata only; never touch models or labels."""
import sys
import json
import hashlib
import collections
import importlib.metadata
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'python_tools'))
from prepare_live_training import validate_annotation
from export_reviewed_streaks import target

captures, totals, sessions, duplicates = [], collections.Counter(), collections.Counter(), {}
for folder in sorted((root / 'training_data/live').iterdir()):
    if not folder.is_dir() or folder.name.startswith('.') or not (folder / 'capture.json').is_file():
        continue
    cap = json.loads((folder / 'capture.json').read_text(encoding='utf-8'))
    ann = json.loads((folder / 'annotations.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line]
    counts, targets, errors = collections.Counter(), [], []
    for row in rows:
        a = ann.get(row['file'], {})
        counts['saved_reviewed'] += a.get('reviewed') is True
        flags = a.get('reviewed_classes')
        counts['ball_reviewed_flag'] += (flags.get('ball') if isinstance(flags, dict) else a.get('reviewed')) is True
        try:
            labels = validate_annotation(a, ('ball',))
            if labels is not None:
                counts['eligible_ball_positive' if labels['ball'] else 'eligible_ball_absent'] += 1
            scene = validate_annotation(a, ('person', 'table'))
            if scene is not None:
                counts['eligible_scene_frames'] += 1
            t = target(a, (cap['width'], cap['height']))
            targets.append(t)
            if t:
                counts['eligible_cnn_' + t['visibility'] + '_frames'] += 1
        except Exception as ex:
            errors.append({'file': row['file'], 'error': str(ex)})
            targets.append(None)
        sourcekey = (cap.get('original_sha256'), row.get('source_frame'))
        duplicates.setdefault(sourcekey, []).append((folder.name, row['file']))
    for i in range(2, len(rows)):
        group, labels = rows[i - 2:i + 1], targets[i - 2:i + 1]
        ids, pts = [r['source_frame'] for r in group], [r['elapsed_ms'] for r in group]
        if (all(labels) and ids == list(range(ids[0], ids[0] + 3))
                and len({str(r.get('burst_id')) for r in group}) == 1
                and all(b > a for a, b in zip(pts, pts[1:]))
                and all(t['visibility'] != 'partial' for t in labels)):
            counts['eligible_cnn_three_frame_windows'] += 1
    totals.update(counts)
    totals['sampled_frames'] += len(rows)
    sessions[cap.get('session_id')] += len(rows)
    captures.append({'clip': folder.name, 'session_id': cap.get('session_id'), 'split': cap.get('split'),
                     'sampled_frames': len(rows), 'capture_origin': cap.get('capture_origin'),
                     'continuity_metadata': {k: v for k, v in cap.items() if any(x in k.lower() for x in ('cut', 'continu', 'duplicate'))},
                     'counts': dict(counts), 'errors': errors,
                     **{n + '_sha256': hashlib.sha256((folder / f).read_bytes()).hexdigest()
                        for n, f in [('annotations', 'annotations.json'), ('capture', 'capture.json'), ('index', 'frames.jsonl')]}})
versions = {}
for name in ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python', 'PyYAML', 'hydra-core', 'omegaconf', 'wandb', 'timm', 'imgaug', 'onnx', 'tensorrt'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
report = {'schema_version': 1,
          'scope': 'Read-only saved-label structural readiness. Label correctness and temporal continuity are not proven. No GPU work, fitting, installation or label mutation.',
          'python': sys.version, 'python_executable': sys.executable, 'dependency_metadata': versions,
          'project_torchvision_overlay': (root / 'artifacts/python_deps/torchvision').is_dir(),
          'frozen_live_splits_exists': (root / 'training_data/live/splits.json').is_file(),
          'totals': dict(totals), 'sessions': dict(sessions), 'unique_source_frames': len(duplicates),
          'repeated_saved_source_frames': {str(k): v for k, v in duplicates.items() if len(v) > 1},
          'captures': captures, 'ready_for_real_training': False,
          'blockers': ['Only one real gameplay session; independent validation/test sessions unavailable',
                       'No frozen live splits manifest',
                       'New no-ball images remain unreviewed; declaration is not saved frame ground truth',
                       'Cut/edit/duplicate history remains unspecified; adjacent source IDs alone do not establish temporal continuity',
                       'No executed BlurBall backward/optimizer-step evidence found in inspected reports; export/parity are forward-only']}
(root / 'artifacts/current_training_readiness.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k not in ('captures', 'repeated_saved_source_frames')}, indent=2))
print(json.dumps([{'clip': r['clip'], 'sampled': r['sampled_frames'], 'counts': r['counts'], 'errors': r['errors']} for r in captures], indent=2))
