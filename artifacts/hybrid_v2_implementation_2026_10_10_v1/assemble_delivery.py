"""Reverify this implementation's evidence and unchanged production artifacts."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python_tools'))
from yolo_ball_experiments import validate_saved_plan


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


folder = 'artifacts/hybrid_v2_implementation_2026_10_10_v1'
prior = load('artifacts/reviewed_training_handoff_delivery_2026_10_10_v1.json')
production = {}
for kind, record in prior['models'].items():
    current = digest(record['path'])
    require(current == record['sha256'], f'Deployed model changed: {kind}')
    production[kind] = dict(path=record['path'], sha256=current, unchanged=True)
backend = 'backend_cpp/build/Release/tt_intelligence_candidate.exe'
require(digest(backend) == prior['backend_sha256'], 'Production native backend changed')
official_path = 'artifacts/paired_ball_official_22_003_primary_v2/metrics.json'
venue_path = folder + '/reviewed_negative_metrics.json'
official, venue = load(official_path), load(venue_path)
runtime = folder + '/eval_paired_ball_candidates.exe'
for report, audit in ((official, official['audit']), (venue, venue['journal_audit'])):
    require(audit['status'] == 'PASS' and audit['runtime_binary_current_file_reverified'], 'Native audit/binary not verified')
    require(audit['runtime_binary_sha256_current_file'] == digest(runtime), 'Archived native evaluator changed')
require(official['audit']['frames_accounted_for'] == 272 and venue['journal_audit']['frames_accounted_for'] == 9164,
        'Complete source coverage missing')
native = {name: load(folder + '/' + name + '.json') for name in ('native_v1_checks', 'native_v2_checks', 'native_cli_checks')}
require(all(value['passed'] for value in native.values()), 'Native verification failed')
student_path = 'artifacts/temporal_ball_student_real_smoke_2026_10_10_v3/report.json'
student = load(student_path)
require(student['passed'] and len(student['steps']) == 2 and not student['checkpoint_saved'], 'Real student smoke failed')
require(all(student[field] is True for field in ('snapshot_reverified_after_optimizer',
    'encoder_weights_and_statistics_unchanged', 'original_checkpoint_unchanged',
    'prototype_and_smoke_source_unchanged', 'selection_helper_source_unchanged')), 'Student provenance check failed')
for field, path in (('prototype_source_sha256', 'python_tools/temporal_ball_student.py'),
                    ('smoke_source_sha256', 'python_tools/smoke_temporal_ball_student.py')):
    require(digest(path) == student[field], 'Student implementation changed after checks')
require(digest('yolo11n.pt') == student['model']['checkpoint_sha256'], 'YOLO encoder checkpoint changed')
plan_path = 'artifacts/yolo_experiment_plan_2026_10_10_v3.json'
plan = validate_saved_plan(load(plan_path))
require(not plan['ready_for_fitting'], 'Data readiness changed unexpectedly')
require(not (ROOT / 'artifacts/yolo_ball_blocked_check_2026_10_10_v3').exists(), 'Unavailable-data fit created output')
tests = {'scorer_and_review': (folder + '/all_scorer_regressions.log', 52),
         'temporal_student': ('artifacts/temporal_ball_student_tests_2026_10_10_v3.log', 15),
         'yolo_experiments': ('artifacts/yolo_ball_experiment_tests_2026_10_10_v3.log', 15)}
for path, count in tests.values():
    text = (ROOT / path).read_text(encoding='utf-8-sig')
    require(re.search(rf'Ran {count} tests? in ', text) and re.search(r'^OK\s*$', text, re.M), 'Recorded regression suite failed')
negative = venue['deduplicated_union_metrics']
counts = {key: value['confirmed_absent_frame_false_positives'] for key, value in negative.items()}
target = min(counts[key] for key in ('standalone_yolo', 'standalone_blurball_causal'))
evidence = [official_path, venue_path, student_path, plan_path, runtime] + [p for p, _ in tests.values()]
evidence += [folder + '/' + name + '.json' for name in native]
report = dict(date='2026-10-10', timezone='Asia/Calcutta', implementation_verified=True,
              scope='Experimental native selector, causal YOLO-feature student and isolated fitting runner',
              deployed_models=production, backend_sha256=digest(backend), backend_unchanged=True,
              native_checks={key: value.get('check_count', len(value.get('checks', []))) for key, value in native.items()},
              python_tests={key: count for key, (_, count) in tests.items()}, real_student_optimizer_steps=2,
              official_development=official['accuracy']['full_recording'],
              reviewed_absent_frames=venue['reviewed_absent_unique_frames_scored'], negative_false_positives=counts,
              negative_gate=dict(maximum_accepted=target, actual=counts['experimental_selector'],
                                 passed=counts['experimental_selector'] <= target),
              ready_for_full_fitting=False, fitting_blockers=plan['blockers'],
              trained_candidate_saved=False, promotion_approved=False, application_default_changed=False,
              live_accuracy_verified=False, live_latency_measured=False, whole_plan_complete=False,
              evidence_sha256={path: digest(path) for path in evidence})
output = ROOT / folder / 'delivery.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({key: report[key] for key in ('implementation_verified', 'native_checks', 'python_tests', 'negative_gate', 'promotion_approved')}, indent=2))
