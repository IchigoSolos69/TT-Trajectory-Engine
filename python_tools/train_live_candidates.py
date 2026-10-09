"""Train separate candidates from reviewed live/match data; never auto-deploy."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/python_deps'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--kind',choices=['ball','scene','both'],default='both')
    parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args()
    audit=json.loads((args.prepared/'audit.json').read_text())
    kinds = ('ball','scene') if args.kind=='both' else (args.kind,)
    counts = audit.get('reviewed_counts_by_kind')
    if counts is None: counts = {kind: audit['reviewed_counts'] for kind in kinds}
    if audit['contains_test_in_training'] or any(kind not in counts or min(counts[kind].values())<30 for kind in kinds):
        raise ValueError('Need at least 30 reviewed frames per independent split; test recordings must stay held out')
    for split in ('train','val','test'):
        ball_frames=[row for row in audit['frames'] if row['split']==split and row.get('kind','ball')=='ball']
        scene_frames=[row for row in audit['frames'] if row['split']==split and (row.get('kind')=='scene' or 'kind' not in row)]
        if args.kind in ('ball','both') and (sum(row['objects']['ball']>0 for row in ball_frames)<5 or sum(row['objects']['ball']==0 for row in ball_frames)<5):
            raise ValueError(f'{split} needs at least five ball-positive and five reviewed absent-ball examples')
        if args.kind in ('scene','both') and sum(row['objects']['person']>0 for row in scene_frames)<5:
            raise ValueError(f'{split} needs real people examples as well as room backgrounds')
    if args.epochs<4:raise ValueError('Use at least four epochs for this training schedule')
    if args.output.exists():raise FileExistsError('Choose a new candidate output directory')
    from ultralytics import YOLO
    for kind in (('ball','scene') if args.kind=='both' else (args.kind,)):
        weights=ROOT/f'artifacts/{"multigame" if kind=="ball" else "scene_detector"}/candidate/weights/best.pt'
        if not weights.is_file():raise FileNotFoundError(weights)
        YOLO(str(weights)).train(data=str(args.prepared/kind/'data.yaml'),epochs=args.epochs,
            imgsz=640,batch=16,device=0,workers=1,optimizer='AdamW',lr0=.0001,lrf=.1,
            freeze=10,cos_lr=True,mosaic=.1,close_mosaic=3,scale=.3,translate=.08,
            hsv_s=.5,hsv_v=.5,seed=42,deterministic=True,patience=5,
            project=str(args.output),name=kind,exist_ok=False)
    print('Candidates trained. Evaluate live test clips and match regressions, then export and check C++ parity before promotion.')
