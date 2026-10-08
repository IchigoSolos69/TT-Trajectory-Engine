"""Resume saved optimizer/EMA state with bounded Windows data-loader memory."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/python_deps'))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    from ultralytics import YOLO
    YOLO(str(args.checkpoint)).train(resume=True,workers=1,device=0)
