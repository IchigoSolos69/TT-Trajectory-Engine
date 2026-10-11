"""Export A/B only for experimental YOLO fitting; C remains sealed."""
import argparse
import json
from pathlib import Path
from yolo_ball_experiments import ROOT,remember,save_plan
from yolo_paired_data import inspect_training_sources,prepare_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--captures",type=Path,default=ROOT/"training_data/live")
    parser.add_argument("--registry",type=Path)
    parser.add_argument("--match-root",type=Path,default=ROOT/"artifacts/multigame")
    args=parser.parse_args()
    dependencies={}
    registry=args.registry or args.captures/"training_sessions.json"
    report=inspect_training_sources(args.captures,registry,args.match_root,
        lambda path,expected=None:remember(dependencies,path,expected))
    output=prepare_records(report,args.output)
    for path,digest in dependencies.items(): remember({},path,digest)
    save_plan(dict(source_sha256=dependencies,test_labels_consumed=False),output/"source_manifest.json")
    print(json.dumps(dict(output=output.as_posix(),strict_data_blockers=report["blockers"],test_exported=False),indent=2))


if __name__=="__main__": main()
