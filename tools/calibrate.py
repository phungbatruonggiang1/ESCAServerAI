"""Score every clip we have and see where normal and anomalous separate.

The threshold that ships with the model (5.6e-05, from metrics_detail.json)
does not match the scale this model actually produces, so it has to be
measured rather than trusted.
"""
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.anomaly import AnomalyDetector
from server.features import FeatureSpec
from server.audio import from_wav, fit_to_length
from pathlib import Path

spec = FeatureSpec()
det = AnomalyDetector(Path(sys.argv[1]), 0.0, spec, threads=2)

def score_all(paths, name):
    out = []
    for p in paths:
        try:
            pcm, sr = from_wav(open(p, 'rb').read())
        except Exception as e:
            print(f'  bo qua {os.path.basename(p)}: {e}'); continue
        if sr != spec.sample_rate or abs(len(pcm) - spec.samples) > spec.sample_rate * 0.1:
            print(f'  bo qua {os.path.basename(p)}: {sr} Hz, {len(pcm)} mau'); continue
        out.append(det.score(fit_to_length(pcm, spec.samples)).score)
    a = np.array(out)
    if len(a):
        print(f'{name:10s} n={len(a):3d}  min={a.min():.5f}  p50={np.median(a):.5f}  '
              f'max={a.max():.5f}  mean={a.mean():.5f}')
    return a

anom = score_all(sorted(glob.glob(sys.argv[2] + '/inference/data/*.wav')), 'LOI')
norm = score_all(sorted(glob.glob(sys.argv[2] + '/inference/record/*.wav'))
                 + sorted(glob.glob(sys.argv[3] + '/ESCA/database/*.wav')), 'BINH THUONG')

if len(anom) and len(norm):
    print()
    print(f'cao nhat cua binh thuong : {norm.max():.6f}')
    print(f'thap nhat cua loi        : {anom.min():.6f}')
    if norm.max() < anom.min():
        th = (norm.max() + anom.min()) / 2
        print(f'=> TACH HOAN TOAN. Nguong de xuat = {th:.6f}')
    else:
        overlap = np.mean(anom < norm.max())
        print(f'=> CHONG LAN: {overlap*100:.0f}% file loi nam duoi muc cao nhat cua binh thuong')
        best, bestf1 = None, -1
        for th in np.unique(np.concatenate([norm, anom])):
            tp = (anom > th).sum(); fp = (norm > th).sum(); fn = (anom <= th).sum()
            f1 = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) else 0
            if f1 > bestf1: bestf1, best = f1, th
        print(f'=> Nguong F1 tot nhat = {best:.6f} (F1 = {bestf1:.3f})')
