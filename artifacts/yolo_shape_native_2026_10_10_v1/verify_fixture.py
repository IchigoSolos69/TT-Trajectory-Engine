"""One-image export/CLI smoke only; not accuracy or live performance."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'artifacts/multigame/candidate/weights/best.pt'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
checkpoint_hash = sha(CHECKPOINT)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
model = YOLO(str(CHECKPOINT)).model.float().cuda().eval()
model.fuse()
records=[]
for name in ['dump960_cpu','dump960_cuda']:
    folder=OUT/name
    contract=json.loads((folder/'contract.json').read_text())
    for key,file in [('input_sha256','input.f32'),('output_sha256','output.f32'),('decoded_bgr_sha256','source.bgr8')]:
        assert sha(folder/file)==contract[key]
    x=np.fromfile(folder/'input.f32',dtype='<f4').reshape(contract['input_shape'])
    native=np.fromfile(folder/'output.f32',dtype='<f4').reshape(contract['output_shape'])
    with torch.inference_mode():
        prediction=model(torch.from_numpy(x).cuda())
        if isinstance(prediction,tuple):prediction=prediction[0]
        reference=prediction.cpu().numpy()
    assert reference.shape==native.shape and np.isfinite(native).all() and np.isfinite(reference).all()
    reference_index=int(reference[0,4].argmax());native_index=int(native[0,4].argmax())
    scale=contract['letterbox']['scale']
    center_error=float(np.linalg.norm(native[0,:2,native_index]-reference[0,:2,reference_index])/scale)
    confidence_error=float(abs(native[0,4,native_index]-reference[0,4,reference_index]))
    passed=center_error<=2 and confidence_error<=.02
    records.append({'dump':name,'preprocessing_revision':contract['preprocessing_revision'],'engine_sha256':contract['engine_sha256'],
                    'native_binary_sha256':contract['native_binary_sha256'],'input_sha256':contract['input_sha256'],
                    'output_shape':contract['output_shape'],'max_raw_coordinate_delta_input_px':float(abs(native[:,:4]-reference[:,:4]).max()),
                    'max_raw_score_delta':float(abs(native[:,4:]-reference[:,4:]).max()),'reference_top_anchor':reference_index,
                    'native_top_anchor':native_index,'top_center_delta_source_px':center_error,'top_score_delta':confidence_error,'passed':passed})
unicode_dir=OUT/'Unicode space 球 v2'
unicode_dir.mkdir(exist_ok=False)
source=unicode_dir/'image 球.png';engine=unicode_dir/'engine 球.engine'
shutil.copyfile(OUT/'export960/yolov11_ttgames.engine',engine)
shutil.copyfile(ROOT/'artifacts/phone_dim_ball_diagnosis/original_reference.png',source)
environment=os.environ.copy()
environment['PATH']=';'.join(['C:/TensorRT/bin','C:/opencv/build/x64/vc16/bin','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64',environment.get('PATH','')])
command=[str(ROOT/'backend_cpp/build/Release/dump_detector_raw.exe'),'--engine',str(engine),'--image',str(source),'--output',str(unicode_dir/'dump 球')]
run=subprocess.run(command,env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace')
(OUT/'unicode_cli.log').write_text(run.stdout+run.stderr,encoding='utf-8')
assert run.returncode==0
contract_path=unicode_dir/'dump 球'/'contract.json';contract=json.loads(contract_path.read_text(encoding='utf-8'))
assert contract['source_image_path']==str(source).replace('\\','/') or Path(contract['source_image_path'])==source
assert contract['engine_sha256']==sha(engine) and contract['source_image_sha256']==sha(source)
before=sha(contract_path)
run_again=subprocess.run(command,env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace')
assert run_again.returncode!=0 and sha(contract_path)==before
assert sha(CHECKPOINT)==checkpoint_hash
report={'scope':'One original frame, actual960FP16 export/rawTensor/UnicodeCLI smoke; not accuracy, no model promotion, no live benchmark',
        'checkpoint_sha256':checkpoint_hash,'criteria':{'top_center_delta_source_px_at_most':2,'top_score_delta_at_most':.02},
        'raw_parity':records,'unicode_image_engine_output_cli_passed':True,'existing_output_refused_unchanged':True,'original_checkpoint_unchanged':True,
        'passed':all(item['passed'] for item in records)}
(OUT/'fixture_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['passed']:raise RuntimeError('Single-frame export smoke exceeded explicit top-candidate parity criteria')
