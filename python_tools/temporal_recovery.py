"""Bounded five-frame ball recovery candidate; no automatic model promotion.

Training uses game_1/game_2 only. Validation uses the existing chronological
game_1 validation range. ROI centers use previous annotations plus seeded jitter:
this is a conditional ROI benchmark, NOT an end-to-end detector benchmark.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import random
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'python_tools/dataset'
SIDE, SIZE, GRID = 192, 128, 32


class TemporalRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(15, 24, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(), nn.Conv2d(48, 64, 3, 1, 1),
            nn.ReLU(), nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.presence = nn.Linear(32, 1)

    def forward(self, images):
        features = self.features(images)
        return torch.cat((self.heatmap(features).flatten(1),
                          self.presence(features.amax((2, 3)))), dim=1)


def status(p):
    if not isinstance(p, dict):
        return 'unknown'
    x, y = p.get('x'), p.get('y')
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return 'unknown'
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError('Invalid annotation')
    if x >= 0 and y >= 0:
        return 'positive'
    return 'negative' if x == y == -1 else 'unknown'


def crop(frame, left, top):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, left), max(0, top), min(w, left+SIDE), min(h, top+SIDE)
    result = np.full((SIDE, SIDE, 3), 114, np.uint8)
    if x2 > x1 and y2 > y1:
        result[y1-top:y2-top, x1-left:x2-left] = frame[y1:y2, x1:x2]
    return cv2.cvtColor(cv2.resize(result, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)


def prepare(output):
    if (output/'dataset.json').exists():
        raise FileExistsError('Prepared data already exists; use a separate output directory')
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    audit = {'conditional_roi_evaluation': True, 'test_sequences_used': False,
             'input': [15, SIZE, SIZE], 'crop_side_pixels': SIDE, 'splits': {}}
    # Deliberately bounded first training experiment. Validation begins after
    # the original training cutoff and its 120-frame exclusion buffer.
    ranges = [('train', 'game_1', 5, 20000, 800), ('train', 'game_2', 5, 20000, 1000),
              ('val', 'game_1', 70999, 79416, 700)]
    arrays = {'train': [], 'val': []}
    targets = {'train': [], 'val': []}
    provenance = {'train': [], 'val': []}
    for split, game, first, last, limit in ranges:
        label_path = DATA/'raw_markup'/game/'ball_markup.json'
        labels = json.loads(label_path.read_text())
        selected = []
        for fid in sorted(map(int, labels)):
            if not first <= fid <= last or status(labels[str(fid)]) == 'unknown':
                continue
            previous = [labels.get(str(fid-d)) for d in range(1, 5)]
            known = next((p for p in previous if status(p) == 'positive'), None)
            if known is not None:
                selected.append((fid, known))
        if len(selected) > limit:
            selected = sorted(rng.sample(selected, limit))
        if not selected:
            raise ValueError(f'No valid temporal samples: {game}/{split}')
        selected = dict(selected)
        needed = {fid-d for fid in selected for d in range(5)}
        cap = cv2.VideoCapture(str(resolve_public_dataset_video(game)))
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, min(needed)-1):
            raise RuntimeError(f'Cannot decode {game}')
        frames = deque(maxlen=5)
        for fid in range(min(needed), max(selected)+1):
            ok = cap.grab()
            if not ok:
                raise RuntimeError(f'Premature video EOF {game}:{fid}')
            if fid not in needed:
                frames.clear()
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise RuntimeError(f'Cannot retrieve {game}:{fid}')
            frames.append((fid, image))
            if fid not in selected:
                continue
            if len(frames) != 5 or frames[0][0] != fid-4:
                raise RuntimeError('Temporal window is not consecutive')
            previous, target = selected[fid], labels[str(fid)]
            cx, cy = previous['x']+rng.randint(-45,45), previous['y']+rng.randint(-45,45)
            centers = [(cx, cy)]
            if fid % 3 == 0:
                centers.append((cx+rng.choice([-300,300]), cy+rng.randint(-100,100)))
            for cx, cy in centers:
                left, top = int(cx-SIDE/2), int(cy-SIDE/2)
                x, y = (target['x']-left)/SIDE, (target['y']-top)/SIDE
                present = status(target) == 'positive' and 0 <= x < 1 and 0 <= y < 1
                # A verified ball outside this crop is a crop-level negative;
                # unknown full-frame labels never become negatives.
                arrays[split].append(np.stack([crop(f, left, top) for _, f in frames]))
                targets[split].append([x if present else 0, y if present else 0, float(present)])
                provenance[split].append({'game': game, 'frame': fid, 'left': left, 'top': top,
                                          'label': 'positive' if present else 'crop_negative'})
        cap.release()
        audit.setdefault('sources', []).append({'split': split, 'game': game, 'first': first,
            'last': last, 'selected_frames': len(selected),
            'markup_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest()})
        print(f'Prepared {game} {split}: {len(selected)} temporal windows', flush=True)
    for split in arrays:
        x, y = np.stack(arrays[split]), np.asarray(targets[split], np.float32)
        np.savez_compressed(output/f'{split}.npz', images=x, targets=y)
        (output/f'{split}_provenance.json').write_text(json.dumps(provenance[split]))
        audit['splits'][split] = {'samples': len(x), 'positive': int(y[:,2].sum())}
    (output/'dataset.json').write_text(json.dumps(audit, indent=2))


class Crops(torch.utils.data.Dataset):
    def __init__(self, path, augment=False):
        data = np.load(path)
        self.images, self.targets, self.augment = data['images'], data['targets'], augment
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        images = self.images[index].copy()
        target = self.targets[index].copy()
        if self.augment:
            if random.random() < .35:
                length = random.choice([3,5,7,9])
                kernel = np.zeros((length,length), np.float32)
                kernel[length//2,:] = 1/length
                for t in range(5):
                    images[t] = cv2.filter2D(images[t], -1, kernel)
            if target[2] and random.random() < .20:
                # Synthetic current-frame occlusion trains a conditional recovery
                # hypothesis. It does not turn presence into a visibility label.
                x, y = int(target[0]*SIZE), int(target[1]*SIZE)
                images[-1,max(0,y-7):min(SIZE,y+8),max(0,x-7):min(SIZE,x+8)] = 114
        tensor = torch.from_numpy(images.transpose(0,3,1,2).reshape(15,SIZE,SIZE)).float()/255
        if self.augment:
            tensor = (tensor*random.uniform(.75,1.25)+random.uniform(-.06,.06)).clamp(0,1)
        return tensor, torch.from_numpy(target)


def evaluate(model, loader, device):
    scores, errors, positives = [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).cpu()
            peak = pred[:,:GRID*GRID].argmax(1)
            coords = torch.stack(((peak%GRID+.5)/GRID,(peak//GRID+.5)/GRID),1)
            errors.extend(((coords-y[:,:2]).norm(dim=1)*SIDE).tolist())
            positives.extend(y[:,2].bool().tolist())
            scores.extend(pred[:,-1].sigmoid().tolist())
    rows = []
    for threshold in [.5,.7,.85,.95]:
        tp = sum(s>=threshold and p and e<=15 for s,p,e in zip(scores,positives,errors))
        fp = sum(s>=threshold and (not p or e>15) for s,p,e in zip(scores,positives,errors))
        fn = sum(p and (s<threshold or e>15) for s,p,e in zip(scores,positives,errors))
        rows.append({'threshold': threshold, 'precision': tp/max(1,tp+fp), 'recall': tp/max(1,tp+fn),
                     'tp': tp, 'fp': fp, 'fn': fn})
    positive_errors = np.asarray([e for e,p in zip(errors,positives) if p])
    return {'thresholds': rows, 'positive_rmse_px': float(np.sqrt(np.mean(positive_errors**2))),
            'positive_median_px': float(np.median(positive_errors)), 'samples': len(scores),
            'scope': 'Prior-annotation-conditioned ROI validation; not end-to-end or live accuracy.'}


def train(output, epochs, initial_checkpoint=None, learning_rate=.001):
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Training requires positive epochs and learning rate')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if (output/'best.pt').exists():
        raise FileExistsError('Candidate exists; use a separate experiment directory')
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4)
    training = torch.utils.data.DataLoader(Crops(output/'train.npz',True),batch_size=32,shuffle=True,num_workers=0)
    validation = torch.utils.data.DataLoader(Crops(output/'val.npz'),batch_size=64,num_workers=0)
    model = TemporalRecovery().cuda()
    if initial_checkpoint:
        model.load_state_dict(torch.load(initial_checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    (output/'training_config.json').write_text(json.dumps({
        'epochs': epochs, 'seed': 42, 'learning_rate': learning_rate,
        'initial_checkpoint_sha256': hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest() if initial_checkpoint else None,
        'optimizer_restored': False, 'selection': 'maximum validation F1 at score 0.5',
        'deployment': 'No automatic promotion'}, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    grid_y,grid_x = torch.meshgrid(torch.arange(GRID,device='cuda')+.5,torch.arange(GRID,device='cuda')+.5,indexing='ij')
    best = -1;history=[]
    for epoch in range(epochs):
        model.train();total=0
        for images,targets in training:
            images,targets=images.cuda(),targets.cuda()
            out=model(images)
            heat=torch.exp(-((grid_x-targets[:,0,None,None]*GRID)**2+(grid_y-targets[:,1,None,None]*GRID)**2)/(2*1.2**2)).flatten(1)
            heat=heat/heat.sum(1,keepdim=True).clamp_min(1e-8)
            positive=targets[:,2]
            location=(-(heat*F.log_softmax(out[:,:GRID*GRID],1)).sum(1)*positive).sum()/positive.sum().clamp_min(1)
            presence=F.binary_cross_entropy_with_logits(out[:,-1],positive)
            loss=location+presence
            optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())
        metrics=evaluate(model,validation,'cuda')
        row={'epoch':epoch+1,'loss':total/len(training),**metrics};history.append(row)
        selected=metrics['thresholds'][0];f1=2*selected['precision']*selected['recall']/max(1e-9,selected['precision']+selected['recall'])
        if f1>best:
            best=f1;torch.save({'state_dict':model.cpu().state_dict(),'epoch':epoch+1,'metrics':metrics},output/'best.pt');model.cuda()
        (output/'training.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({'epoch':epoch+1,'loss':row['loss'],'f1_at_05':f1,'median_px':metrics['positive_median_px']}),flush=True)
    checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    (output/'validation.json').write_text(json.dumps(checkpoint['metrics'],indent=2))


class StressCrops(Crops):
    def __init__(self, path, mask_current=False, repeat_current=False):
        super().__init__(path)
        self.mask_current,self.repeat_current=mask_current,repeat_current
    def __getitem__(self,index):
        tensor,target=super().__getitem__(index)
        if self.mask_current and target[2]:
            x,y=int(target[0]*SIZE),int(target[1]*SIZE)
            tensor[-3:,max(0,y-7):min(SIZE,y+8),max(0,x-7):min(SIZE,x+8)]=114/255
        if self.repeat_current:
            tensor=tensor[-3:].repeat(5,1,1)
        return tensor,target


def validate_export(output):
    import onnx
    checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    model=TemporalRecovery().eval();model.load_state_dict(checkpoint['state_dict'])
    report={'epoch':checkpoint['epoch'],'experiments':{},'promotion':'NOT_APPROVED',
            'scope':'Chronological, prior-annotation-conditioned ROI validation. Synthetic masking is not natural occlusion. No live test or end-to-end detector comparison.'}
    for name,mask,repeat in [('five_frames',False,False),('repeated_current',False,True),
                             ('masked_current',True,False),('masked_repeated_current',True,True)]:
        loader=torch.utils.data.DataLoader(StressCrops(output/'val.npz',mask,repeat),batch_size=64,num_workers=0)
        report['experiments'][name]=evaluate(model.cuda(),loader,'cuda')
    model.cpu()
    destination=output/'temporal.onnx'
    if destination.exists():raise FileExistsError(destination)
    torch.onnx.export(model,torch.zeros(1,15,SIZE,SIZE),destination,input_names=['frames'],output_names=['recovery'],opset_version=17,dynamo=False)
    graph=onnx.load(destination);onnx.checker.check_model(graph)
    shapes=[[d.dim_value for d in value.type.tensor_type.shape.dim] for value in list(graph.graph.input)+list(graph.graph.output)]
    if shapes!=[[1,15,SIZE,SIZE],[1,GRID*GRID+1]]:raise ValueError(f'Unexpected temporal shapes {shapes}')
    examples=Crops(output/'val.npz');inputs=torch.stack([examples[i][0] for i in range(32)])
    with torch.inference_mode():predictions=model(inputs).numpy()
    inputs.numpy().tofile(output/'parity_input.bin');predictions.tofile(output/'parity_pytorch.bin')
    report['onnx_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
    report['checkpoint_sha256']=hashlib.sha256((output/'best.pt').read_bytes()).hexdigest()
    report['input']=shapes[0];report['output']=shapes[1];report['precision']='FP32'
    (output/'recovery_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','validate-export'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--initial-checkpoint',type=Path)
    parser.add_argument('--learning-rate',type=float,default=.001)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='train':train(args.output,args.epochs,args.initial_checkpoint,args.learning_rate)
    else:validate_export(args.output)
