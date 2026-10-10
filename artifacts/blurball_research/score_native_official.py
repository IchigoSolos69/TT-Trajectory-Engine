import json
import sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python_tools'))
from evaluate_blurball_candidate import select
from score_blurball_comparison import score
research = root / 'artifacts/blurball_research'
directory = research / 'native_official_22_003'
windows = [json.loads(line) for line in (directory / 'windows.jsonl').read_text().splitlines()]
baseline = [json.loads(line) for line in (research / 'official_22_003/comparison/predictions.jsonl').read_text().splitlines()]
all_candidates = [[] for _ in baseline]
previous_causal = None
for row in baseline:
    row['blurball_causal'] = None
    row['blurball_offline'] = None
for window in windows:
    for channel, (fid, candidates) in enumerate(zip(window['source_frame_ids'],window['channels'])):
        for candidate in candidates:
            candidate['window_start_frame'] = window['source_frame_ids'][0]
            candidate['output_channel'] = channel
            candidate['lookahead_frames'] = 2-channel
        all_candidates[fid-1].extend(candidates)
        if channel == 2:
            previous_causal = select(candidates, previous_causal, 300.)
            baseline[fid-1]['blurball_causal'] = previous_causal
previous_offline = None
for row,candidates in zip(baseline,all_candidates):
    previous_offline = select(candidates,previous_offline,300.)
    row['blurball_offline'] = previous_offline
predictions = directory/'predictions.jsonl'
predictions.write_text(''.join(json.dumps(row)+'\n' for row in baseline))
report = score(predictions,research/'official_22_003/labels.csv')
report['scope'] = 'Native C++ BlurBall heatmap components on one official 272-frame clip, upstream source-mass/300-pixel selection; saved Python YOLO baseline reused; not live or broad-domain acceptance.'
report['native_manifest'] = json.loads((directory/'manifest.json').read_text())
report['native_stage_timing_offline_multi_pass'] = {
    key:{'median_ms':float(np.median([r[key] for r in windows])), 'p95_ms':float(np.percentile([r[key] for r in windows],95))}
    for key in ['preprocess_ms','h2d_ms','gpu_ms','d2h_ms','decode_ms','wall_ms']}
(directory/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'full_recording':report['full_recording'], 'timings':report['native_stage_timing_offline_multi_pass']},indent=2))
