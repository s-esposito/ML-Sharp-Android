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
import onnxruntime as ort


def sess(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


def describe(name, s):
    print(f"\n[{name}]")
    for i in s.get_inputs():
        print(f"  in  {i.name:70s} {i.shape} {i.type}")
    for o in s.get_outputs():
        print(f"  out {o.name:70s} {o.shape} {o.type}")


def metrics(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    d = np.abs(a - b)
    return cos, float(d.max()), float(d.mean())


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


def check_split(d, rng):
    full = sess(d / 'rest.onnx')
    feeds = rand_inputs(full, rng)
    t = time.time()
    ref = dict(zip([o.name for o in full.get_outputs()], full.run(None, feeds)))
    print(f"\nrest.onnx full run {time.time()-t:.1f}s")
    del full

    pool = dict(feeds)
    for seg in ('a', 'b', 'c'):
        s = sess(d / f'rest_seg_{seg}.onnx')
        need = {i.name: pool[i.name] for i in s.get_inputs()}
        t = time.time()
        outs = s.run(None, need)
        print(f"rest_seg_{seg} run {time.time()-t:.1f}s, {len(outs)} outputs")
        pool.update(zip([o.name for o in s.get_outputs()], outs))
        del s
    ok = True
    for k, v in ref.items():
        cos, mx, mean = metrics(v, pool[k])
        print(f"  split vs full  {k:10s} cos={cos:.8f} max|d|={mx:.3e} mean|d|={mean:.3e}")
        ok &= cos > 0.99999
    return ok


def check_torch(d, sharp_root, rng):
    import torch
    sys.path.insert(0, str(Path(sharp_root) / 'src'))
    from sharp.models import create_predictor
    from sharp.models.params import PredictorParams
    pred = create_predictor(PredictorParams())
    sd = torch.load(Path(sharp_root) / 'sharp_2572gikvuh.pt', map_location='cpu', weights_only=False)
    pred.load_state_dict(sd, strict=False)
    pred.eval()
    enc = pred.monodepth_model.monodepth_predictor.encoder
    x = rng.uniform(-1, 1, (1, 3, 384, 384)).astype(np.float32)
    ok = True
    with torch.no_grad():
        out, interm = enc.patch_encoder(torch.from_numpy(x))
        ref = {'patch_features': out.numpy(),
               'latent0': enc.patch_encoder.reshape_feature(interm[5]).numpy(),
               'latent1': enc.patch_encoder.reshape_feature(interm[11]).numpy()}
        ie_ref = enc.image_encoder(torch.from_numpy(x))[0].numpy()
    s = sess(d / 'patch_encoder.onnx')
    got = dict(zip([o.name for o in s.get_outputs()], s.run(None, {'patch_1': x})))
    del s
    for k in ref:
        cos, mx, mean = metrics(ref[k], got[k])
        print(f"  torch vs onnx  pe.{k:15s} cos={cos:.8f} max|d|={mx:.3e} mean|d|={mean:.3e}")
        ok &= cos > 0.9999
    s = sess(d / 'image_encoder.onnx')
    got = s.run(None, {'image_1': x})[0]
    cos, mx, mean = metrics(ie_ref, got)
    print(f"  torch vs onnx  ie.image_features  cos={cos:.8f} max|d|={mx:.3e} mean|d|={mean:.3e}")
    ok &= cos > 0.9999
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx_dir', default='output/onnx')
    ap.add_argument('--torch', action='store_true')
    ap.add_argument('--sharp_root', default=None)
    ap.add_argument('--skip_split', action='store_true')
    a = ap.parse_args()
    d = Path(a.onnx_dir)
    rng = np.random.default_rng(0)
    for n in ('patch_encoder', 'image_encoder', 'rest_seg_a', 'rest_seg_b', 'rest_seg_c'):
        s = sess(d / f'{n}.onnx')
        describe(n, s)
        del s
    ok = True
    if not a.skip_split:
        ok &= check_split(d, rng)
    if a.torch:
        ok &= check_torch(d, a.sharp_root, rng)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
