#!/usr/bin/env python3
"""verify_onnx.py — desktop sanity checks for the exported SHARP ONNX set.

  1. every model loads in onnxruntime; prints input/output names + shapes
  2. split chain rest_seg_a -> b -> c reproduces the monolithic rest.onnx
  3. (--torch) patch/image encoder ONNX vs the original PyTorch modules

Usage:
  python tools/verify_onnx.py --onnx_dir output/onnx [--torch --sharp_root ~/Codebase/ml-sharp]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_outputs import ROOT, make_session, metrics  # noqa: E402

SEGMENTS = ('patch_encoder', 'image_encoder', 'rest_seg_a', 'rest_seg_b', 'rest_seg_c')


def describe(name, s):
    print(f"\n[{name}]")
    for i in s.get_inputs():
        print(f"  in  {i.name:70s} {i.shape} {i.type}")
    for o in s.get_outputs():
        print(f"  out {o.name:70s} {o.shape} {o.type}")


def check(label, ref, got, min_cos):
    cos, mx, mean, _ = metrics(ref, got)
    print(f"  {label:32s} cos={cos:.8f} max|d|={mx:.3e} mean|d|={mean:.3e}")
    return cos >= min_cos


def rand_inputs(s, rng):
    feeds = {}
    for i in s.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if i.name == 'disparity_factor':
            feeds[i.name] = np.array([1.38], dtype=np.float32)
        elif i.name == 'image':
            feeds[i.name] = rng.random(shape, dtype=np.float32)
        else:
            feeds[i.name] = rng.standard_normal(shape, dtype=np.float32)
    return feeds


def run(s, feeds):
    want = {i.name for i in s.get_inputs()}
    outs = s.run(None, {k: v for k, v in feeds.items() if k in want})
    return dict(zip([o.name for o in s.get_outputs()], outs))


def check_split(d, rng):
    full = make_session(d / 'rest.onnx')
    pool = rand_inputs(full, rng)
    t = time.time()
    ref = run(full, pool)
    print(f"\nrest.onnx full run {time.time()-t:.1f}s")
    del full
    for seg in ('a', 'b', 'c'):
        s = make_session(d / f'rest_seg_{seg}.onnx')
        t = time.time()
        out = run(s, pool)
        print(f"rest_seg_{seg} run {time.time()-t:.1f}s, {len(out)} outputs")
        pool.update(out)
        del s
    return all([check(f"split vs full  {k}", v, pool[k], 0.99999) for k, v in ref.items()])


def check_torch(d, sharp_root, rng):
    import torch
    sys.path.insert(0, str(ROOT))
    import build_rest_pipeline as brp
    if sharp_root:
        brp.SHARP_ROOT = Path(sharp_root).expanduser().resolve()
    brp._require_src()
    pred = brp._load_predictor()
    enc = pred.monodepth_model.monodepth_predictor.encoder
    x = rng.uniform(-1, 1, (1, 3, 384, 384)).astype(np.float32)
    with torch.no_grad():
        out, interm = enc.patch_encoder(torch.from_numpy(x))
        ref = {'patch_features': out.numpy(),
               'latent0': enc.patch_encoder.reshape_feature(interm[5]).numpy(),
               'latent1': enc.patch_encoder.reshape_feature(interm[11]).numpy()}
        ie_ref = enc.image_encoder(torch.from_numpy(x))[0].numpy()
    got = run(make_session(d / 'patch_encoder.onnx'), {'patch_1': x})
    ok = all([check(f"torch vs onnx  pe.{k}", ref[k], got[k], 0.9999) for k in ref])
    got = run(make_session(d / 'image_encoder.onnx'), {'image_1': x})['image_features']
    return check("torch vs onnx  ie.image_features", ie_ref, got, 0.9999) and ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx_dir', default=str(ROOT / 'output' / 'onnx'))
    ap.add_argument('--torch', action='store_true')
    ap.add_argument('--sharp_root', default=None)
    ap.add_argument('--skip_split', action='store_true')
    a = ap.parse_args()
    d = Path(a.onnx_dir)
    rng = np.random.default_rng(0)
    for n in SEGMENTS:
        describe(n, make_session(d / f'{n}.onnx'))
    ok = True
    if not a.skip_split:
        ok &= check_split(d, rng)
    if a.torch:
        ok &= check_torch(d, a.sharp_root, rng)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
