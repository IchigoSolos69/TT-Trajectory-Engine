"""Summarize native regression evidence without assigning any new labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from prepare_live_training import validate_annotation


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(output):
    root = Path(__file__).resolve().parents[1]
    before = read(root/'artifacts/scene_detector/test2.json')
    after = read(output/'test2_scene_after.json')
    if before != after:
        raise RuntimeError('Scene regression metrics changed; inspect native predictions before promotion')
    raw = [json.loads(line) for line in (output/'authority_default/frames.jsonl').read_text().splitlines()]
    gated = [json.loads(line) for line in (output/'authority_experimental/frames.jsonl').read_text().splitlines()]
    blocked = [r['frame_id'] for r in gated if r['intelligence']['trajectory'][-1]['state']=='REJECTED' and raw[r['frame_id']-1]['ball']]
    if not blocked:
        raise RuntimeError('No rejected raw proposal exercised the authority fix')
    ball = next(r for r in csv.DictReader((output/'ball_before.csv').open()) if r['frame']=='1848')
    captures=[]
    for clip in sorted((root/'training_data/live').glob('phone_*')):
        manifest=read(clip/'capture.json')
        annotations=read(clip/'annotations.json')
        reviewed=positive=negative=excluded=0
        for annotation in annotations.values():
            if annotation.get('reviewed_classes',{}).get('ball',annotation.get('reviewed')) is not True:
                continue
            reviewed+=1
            labels=validate_annotation(annotation,('ball',))
            if labels is None: excluded+=1
            elif labels['ball']: positive+=1
            else: negative+=1
        captures.append({'clip':clip.name,'source':Path(manifest['original_path']).name,'session':manifest['session_id'],
                         'reviewed_ball_frames':reviewed,'usable_positive_frames':positive,'usable_negative_frames':negative,
                         'excluded_ambiguous_frames':excluded,'annotation_sha256':hashlib.sha256((clip/'annotations.json').read_bytes()).hexdigest()})
    production=read(output/'production_runtime.json')
    # Completion can follow the selected-frame acceptance packet; read the
    # final journal marker rather than claiming full coverage from a prefix.
    completion=read(root/'artifacts/analyses'/production['analysis_id']/'completion.json')
    result={'scope':'Offline native detection/data-authority diagnostics and browser smoke test; not live accuracy or performance benchmarks.',
            'ball_accuracy_resolved':False,'models_changed':False,
            'reported_frame':{'frame':1848,'native_ball_score':float(ball['confidence']),
                              'native_ball_center_px':[float(ball['x']),float(ball['y'])],
                              'issue':'User-reported ceiling feature false positive, still present in deployed model output'},
            'scene_regression':{'before':before,'after':after,'unchanged':True},
            'rejected_raw_bypass_cases_verified':len(blocked),
            'production_complete_coverage':completion,'captures':captures,
            'live_training_totals':{key:sum(c[key] for c in captures) for key in ('reviewed_ball_frames','usable_positive_frames','usable_negative_frames','excluded_ambiguous_frames')},
            'independent_session_count':len({c['session'] for c in captures}),
            'remaining':['Reviewed difficult-ball positives and true absent-ball negatives',
                         'Separate validation/test sessions; the ten supplied recordings are one session',
                         'Venue-trained candidate, native export parity and fixed precision/recall promotion gates',
                         'Independent grey-table landmarks/calibration',
                         'Native MOV true source-PTS/seek acceptance']}
    (output/'verification_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('reported_frame','rejected_raw_bypass_cases_verified','production_complete_coverage','live_training_totals','independent_session_count')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
