"""Diagnose a reported false positive; hints remain unreviewed and out of training."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from compare_lighting_filters import Detector, variants


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Cannot save diagnostic image: {path}')


def main(args):
    if args.frame < 1:
        raise ValueError('Source frame must be a positive 1-based ordinal')
    args.output.mkdir(parents=True, exist_ok=False)
    model = Detector(args.engine)
    video = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not video.isOpened(): raise RuntimeError('Cannot decode recording')
    wanted = set(range(max(1, args.frame - 5), args.frame + 6))
    rows, images = [], []
    try:
        for fid in range(1, max(wanted) + 1):
            if not video.grab(): raise RuntimeError(f'Decode ended before frame {fid}')
            if fid not in wanted: continue
            ok, frame = video.retrieve()
            if not ok: raise RuntimeError(f'Cannot retrieve frame {fid}')
            candidates = model.infer_candidates(frame)
            panels, alternate = [], {}
            if fid == args.frame:
                hx, hy = args.hint
                if not 0 <= hx < frame.shape[1] or not 0 <= hy < frame.shape[0]:
                    raise ValueError('Screenshot hint is outside the source image')
                changes, quality = variants(frame)
                alternate = {name: model.infer_candidates(image) for name, image in changes.items()}
                # A fixed overlapping grid tests resolution without using the
                # user's marked ball position to choose inference crops.
                tiled = []
                for y in (0, 440):
                    for x in (0, 480, 960):
                        for proposal in model.infer_candidates(frame[y:y+640, x:x+960], limit=8):
                            proposal = dict(proposal)
                            proposal['center_px'] = [proposal['center_px'][0]+x, proposal['center_px'][1]+y]
                            bx, by, bw, bh = proposal['box_px']; proposal['box_px'] = [bx+x, by+y, bw,bh]
                            proposal['tile_origin_px'] = [x,y]
                            tiled.append(proposal)
                alternate['fixed_grid_960x640'] = sorted(tiled,key=lambda p:p['score'],reverse=True)
                write_image(args.output/'original_reference.png',frame)
                # Context crop around the user's screenshot hint, for inspection
                # only; it is NEVER used to select inference regions or labels.
                hx, hy = args.hint; crop = frame[max(0,hy-70):hy+70,max(0,hx-100):hx+100]
                write_image(args.output/'hint_context.png',cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST))
                for name, proposals in alternate.items():
                    panel = cv2.resize(frame,(768,432))
                    for proposal in proposals:
                        if not proposal['accepted_at_0_25']: continue
                        x,y=proposal['center_px'];cv2.circle(panel,(round(x*.4),round(y*.4)),7,(0,180,255),2)
                    cv2.putText(panel,name,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
                    panels.append(panel)
                panels.append(np.zeros_like(panels[0]))
                write_image(args.output/'proposal_comparison.jpg',np.vstack([np.hstack(panels[i:i+2]) for i in (0,2,4)]))
            panel = cv2.resize(frame,(640,360))
            best = candidates[0]
            if best['accepted_at_0_25']:
                x,y = best['center_px'];cv2.circle(panel,(round(x/3),round(y/3)),5,(0,180,255),2)
            cv2.putText(panel,f"source {fid}: {best['score']:.3f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
            images.append(panel)
            hx,hy=args.hint
            near = [p for p in candidates if np.hypot(p['center_px'][0]-hx,p['center_px'][1]-hy)<50]
            rows.append({'frame_id':fid,'source_pts_ms':video.get(cv2.CAP_PROP_POS_MSEC), 'full_frame':candidates,
                         'proposals_near_approximate_screenshot_hint':near,'alternatives':alternate})
            print(f"source frame {fid}: top score {best['score']:.3f}, center {best['center_px']}",flush=True)
    finally: video.release()
    while len(images) % 3:
        images.append(np.zeros_like(images[0]))
    write_image(args.output/'temporal_contact_sheet.jpg',np.vstack([np.hstack(images[i:i+3]) for i in range(0,len(images),3)]))
    result={'video':str(args.video),'engine_sha256':hashlib.sha256(args.engine.read_bytes()).hexdigest(),
            'approximate_user_screenshot_hint_px':args.hint,'hint_reviewed':False,
            'scope':'Sequential offline TensorRT diagnostics; screenshot hint is approximate, not ground truth. No filter/model/threshold promotion.',
            'frames':rows}
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path);p.add_argument('engine',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--frame',type=int,default=1685);p.add_argument('--hint',type=int,nargs=2,default=[1092,582])
    main(p.parse_args())
