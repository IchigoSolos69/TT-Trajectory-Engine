"""Score native scene detections against untouched test_2 segmentation masks."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,r-x)*max(0,bottom-y)
    return inter/max(1e-9,a[2]*a[3]+b[2]*b[3]-inter)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('masks',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--predictions',type=Path)
    args=parser.parse_args()
    masks=sorted(args.masks.glob('*.png'),key=lambda p:int(p.stem))
    if not masks:raise ValueError('No annotated masks')
    if not args.predictions:
        args.output.write_text('\n'.join(p.stem for p in masks)+'\n')
    else:
        rows={}
        for row in csv.DictReader(args.predictions.open()):rows.setdefault(int(row['frame']),[]).append(row)
        counts={0:dict(tp=0,fp=0,fn=0),1:dict(tp=0,fp=0,fn=0)}
        quad_scores=[]
        for path in masks:
            fid=int(path.stem)
            if fid not in rows:raise ValueError(f'Missing inference: {fid}')
            mask=cv2.imread(str(path));h,w=mask.shape[:2]
            for cls,channel in ((0,1),(1,2)):
                _,_,stats,_=cv2.connectedComponentsWithStats((mask[:,:,channel]>75).astype('uint8'))
                gt=[(x/w,y/h,bw/w,bh/h) for x,y,bw,bh,area in stats[1:] if area>=(30 if cls==0 else 200)]
                pred=sorted([r for r in rows[fid] if int(r['class'])==cls and float(r['confidence'])>=.4],key=lambda r:-float(r['confidence']))
                matched=set()
                for r in pred:
                    box=tuple(float(r[k]) for k in ('x','y','w','h'))
                    choices=[(iou(box,b),i) for i,b in enumerate(gt) if i not in matched]
                    overlap,index=max(choices,default=(0,-1))
                    if overlap>=.5:counts[cls]['tp']+=1;matched.add(index)
                    else:counts[cls]['fp']+=1
                counts[cls]['fn']+=len(gt)-len(matched)
            quad=next((r for r in rows[fid] if float(r['quad_x0'])>=0),None)
            if quad:
                predicted=np.zeros((h,w),dtype='uint8')
                pts=np.array([[float(quad[f'quad_x{i}'])*w,float(quad[f'quad_y{i}'])*h] for i in range(4)],dtype='int32')
                cv2.fillConvexPoly(predicted,pts,1)
                truth=mask[:,:,2]>75
                quad_scores.append(float(np.logical_and(predicted,truth).sum()/max(1,np.logical_or(predicted,truth).sum())))
        for c in counts.values():
            c['precision']=c['tp']/max(1,c['tp']+c['fp']);c['recall']=c['tp']/max(1,c['tp']+c['fn'])
        result={'frames':len(masks),'confidence_threshold':.4,'iou_threshold':.5,'person':counts[0],'table':counts[1],
                'table_quad_frames':len(quad_scores),'table_quad_mean_mask_iou':sum(quad_scores)/max(1,len(quad_scores)),
                'limitation':'People include officials; table quad is color/contour refinement of a learned box. Test_2 is dataset footage, not a live camera evaluation.'}
        args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
