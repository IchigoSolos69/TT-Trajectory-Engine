"""Score native image-plane covariance against known annotations; unknowns excluded.

This does not validate table-plane covariance or true 3D reconstruction.
Use a separate run with injected detector misses to test recovery explicitly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def score(samples, labels):
    groups = {}
    missing = false_tracks = unknown = 0
    for s in samples:
        gt = labels.get(str(s['frame_id']))
        if not gt or gt.get('x', -1) < 0 or gt.get('y', -1) < 0:
            if gt and gt.get('x') == -1 and gt.get('y') == -1:
                false_tracks += s['state'] != 'LOST'
            else:
                unknown += 1
            continue
        if s['state'] == 'LOST':
            missing += 1
            continue
        x, y = s['pixel']
        dx, dy = x-gt['x'], y-gt['y']
        a, b, _, c = s['pixel_covariance']
        determinant = a*c-b*b
        d2 = (c*dx*dx-2*b*dx*dy+a*dy*dy)/determinant if determinant > 0 else math.inf
        group = groups.setdefault(s['state'], {'errors': [], 'covered': 0, 'localized': 0})
        error = math.hypot(dx, dy)
        group['errors'].append(error)
        group['covered'] += d2 <= 5.991
        group['localized'] += error <= 15
    result = {}
    for state, g in groups.items():
        errors = sorted(g['errors'])
        result[state] = {'count': len(errors), 'rmse_px': math.sqrt(sum(e*e for e in errors)/len(errors)),
                         'median_px': errors[len(errors)//2], 'p95_px': errors[min(len(errors)-1, int(len(errors)*.95))],
                         'localized_within_15px': g['localized']/len(errors), 'model_95_region_coverage': g['covered']/len(errors)}
    return {'by_state': result, 'known_positive_lost': missing, 'known_absent_with_track': false_tracks,
            'unknown_excluded': unknown, 'warning': 'Image-plane only; empirical coverage is not guaranteed. Sparse annotations; no live-camera accuracy claim.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', type=Path)
    p.add_argument('markup', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--sweep', action='store_true', help='Score every explicitly recorded parameter configuration without choosing on test footage')
    a = p.parse_args()
    labels = json.loads(a.markup.read_text())
    if a.sweep:
        groups = {}
        with a.samples.open() as stream:
            for line in stream:
                value = json.loads(line)
                key = (value['acceleration_sigma'], value['measurement_sigma_scale'])
                groups.setdefault(key, []).append(value['sample'])
        report = {'configurations': [{'acceleration_sigma': key[0], 'measurement_sigma_scale': key[1], **score(samples, labels)} for key, samples in groups.items()]}
    else:
        report = score([json.loads(line) for line in a.samples.read_text().splitlines()], labels)
    report['samples_sha256'] = hashlib.sha256(a.samples.read_bytes()).hexdigest()
    report['markup_sha256'] = hashlib.sha256(a.markup.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
