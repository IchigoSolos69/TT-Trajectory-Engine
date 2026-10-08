"""Compare native C++ TensorRT and PyTorch temporal outputs on identical tensors."""
import argparse
from array import array
import hashlib
import json
import math


def verify(directory):
    def read(name):
        data=array('f');data.frombytes((directory/name).read_bytes())
        if any(not math.isfinite(v) for v in data):
            raise ValueError(f'Nonfinite values in {name}')
        return data
    reference,native=read('parity_pytorch.bin'),read('parity_native.bin')
    if len(reference)!=len(native) or not reference or len(reference)%1025:
        raise ValueError('Parity outputs have inconsistent or incomplete shapes')
    confidence_delta=0.;position_delta=0.;matches=0
    sigmoid=lambda x:1/(1+math.exp(-max(-80.,min(80.,x))))
    for offset in range(0,len(reference),1025):
        a,b=reference[offset:offset+1024],native[offset:offset+1024]
        ia,ib=max(range(1024),key=a.__getitem__),max(range(1024),key=b.__getitem__)
        matches+=ia==ib
        position_delta=max(position_delta,math.hypot(ia%32-ib%32,ia//32-ib//32)*6)
        confidence_delta=max(confidence_delta,abs(sigmoid(reference[offset+1024])-sigmoid(native[offset+1024])))
    max_logit=max(abs(a-b) for a,b in zip(reference,native))
    report={'samples':len(reference)//1025,'max_logit_error':max_logit,
            'max_presence_score_error':confidence_delta,'max_localization_delta_source_px':position_delta,
            'argmax_matches':matches,'scope':'Fixed ROI tensor numerical parity, not tracking or live accuracy.',
            'engine_sha256':hashlib.sha256((directory/'temporal.engine').read_bytes()).hexdigest(),
            'input_sha256':hashlib.sha256((directory/'parity_input.bin').read_bytes()).hexdigest()}
    report['passed']=max_logit<.05 and confidence_delta<.005 and position_delta<=6
    return report


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=verify(args.directory)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Native temporal parity failed')
