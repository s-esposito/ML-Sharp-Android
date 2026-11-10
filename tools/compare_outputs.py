#!/usr/bin/env python3
"""compare_outputs.py — check on-device SHARP outputs against a desktop FP32 reference.

The reference is the FP32 ONNX set run in onnxruntime on the desktop, plus the app's
own C core built for the host (tools/host_post). Comparisons:

  per-model (isolated): device inputs of each model -> desktop model -> compare with the
      device outputs. Pins error to one model (PE, IE, REST chain, C post).
  end-to-end: device image.raw (same preprocessed input) -> full desktop pipeline ->
      compare disparity, delta and the final .ply (Gaussian count + per-attribute error).

Device dumps come from Settings > "Dump model outputs" (see PORTING_NOTES.md for the
adb pull command). A desktop-only reference run (no device) is also possible:

  python tools/compare_outputs.py --dump dumps/<ts> --onnx_dir output/onnx
  python tools/compare_outputs.py --reference_image data/teaser.jpg --out ref_run/   # make a dump-like dir

Metrics: cosine similarity, max |d|, mean |d| (float64). Exit 1 if any check fails its
threshold (FP32 backends: cos >= 0.999 on model outputs; quantised HTP: --quantized).
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
HOST_POST = ROOT / 'tools' / 'host_post' / 'host_post'

ONNX_NAMES = {
    'pe': ['pe.onnx', 'patch_encoder.onnx'],
    'ie': ['ie.onnx', 'image_encoder.onnx'],
    'rest_a': ['rest_a.onnx', 'rest_seg_a.onnx'],
    'rest_b': ['rest_b.onnx', 'rest_seg_b.onnx'],
    'rest_c': ['rest_c.onnx', 'rest_seg_c.onnx'],
}
PE_OUTS = ['patch_features', 'latent0', 'latent1']
# PE patches dumped by the app (PipelineManager.DUMP_PATCHES): first x0 patch, centre x1 patch, x2
DUMP_PATCHES = [0, 29, 34]
MERGED = ['x_latent0', 'x_latent1', 'x0_feat', 'x1_feat', 'x2_feat', 'x_lowres_feat']

RESULTS = []


# ---------------------------------------------------------------- helpers

def load_raw(path, shape=None):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape) if shape is not None else a


def metrics(ref, got):
    a = np.asarray(ref, dtype=np.float64).ravel()
    b = np.asarray(got, dtype=np.float64).ravel()
    if a.size != b.size:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float('nan')
    d = np.abs(a - b)
    return cos, float(d.max()), float(d.mean()), int(np.isnan(b).sum() + np.isinf(b).sum())


def report(group, name, ref, got, min_cos):
    m = metrics(ref, got)
    if m is None:
        print(f"  [{group:8s}] {name:34s} SIZE MISMATCH ref={np.size(ref)} got={np.size(got)}  FAIL")
        RESULTS.append((group, name, None, False))
        return
    cos, mx, mean, bad = m
    ok = (cos >= min_cos) and bad == 0
    print(f"  [{group:8s}] {name:34s} cos={cos:.6f} max|d|={mx:.3e} mean|d|={mean:.3e}"
          f"{' nonfinite=' + str(bad) if bad else ''}  {'ok' if ok else 'FAIL'}")
    RESULTS.append((group, name, m, ok))


def make_session(path, threads=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


class Models:
    def __init__(self, onnx_dir, threads):
        self.dir = Path(onnx_dir)
        self.threads = threads
        self.cache = {}

    def path(self, key):
        for n in ONNX_NAMES[key]:
            p = self.dir / n
            if p.exists():
                return p
        sys.exit(f"missing ONNX for {key} in {self.dir}")

    def sess(self, key):
        # keep at most one session resident (encoders are 1.2 GB each)
        if key not in self.cache:
            self.cache.clear()
            self.cache[key] = make_session(self.path(key), self.threads)
        return self.cache[key]

    def run(self, key, feeds):
        s = self.sess(key)
        want = {i.name for i in s.get_inputs()}
        outs = s.run(None, {k: v for k, v in feeds.items() if k in want})
        return dict(zip([o.name for o in s.get_outputs()], outs))


def host(*args):
    if not HOST_POST.exists():
        sys.exit(f"{HOST_POST} not built: run tools/host_post/build.sh")
    r = subprocess.run([str(HOST_POST), *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"host_post {args[0]} failed rc={r.returncode}: {r.stderr[-500:]}")
    return r.stdout


def read_ply(path):
    """Vertex records (structured array) of the app's binary PLY."""
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header += line
        text = header.decode('ascii')
        n, props, in_vertex = 0, [], False
        for ln in text.splitlines():
            t = ln.split()
            if t[:2] == ['element', 'vertex']:
                n, in_vertex = int(t[2]), True
            elif t and t[0] == 'element':
                in_vertex = False
            elif in_vertex and t[0] == 'property':
                props.append((t[2], '<f4'))
        v = np.fromfile(f, dtype=np.dtype(props), count=n)
    return v


def compare_ply(group, ref_path, got_path, min_cos):
    ref = read_ply(ref_path)
    got = read_ply(got_path)
    print(f"  [{group:8s}] gaussians ref={len(ref):,} device={len(got):,}"
          f"  {'ok' if len(ref) == len(got) else 'FAIL'}")
    RESULTS.append((group, 'gaussian_count', None, len(ref) == len(got)))
    if len(ref) != len(got):
        return
    for name in ref.dtype.names:
        report(group, f"ply.{name}", ref[name], got[name], min_cos)


# ---------------------------------------------------------------- desktop pipeline

def run_desktop(models, image_raw, fpx, ow, oh, work):
    """Full desktop pipeline in the app's on-device file layout. Returns timings."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    t = {}
    if Path(image_raw).resolve() != (work / 'image.raw').resolve():
        shutil.copy(image_raw, work / 'image.raw')
    t0 = time.time(); host('pre', work / 'image.raw', work); t['pre'] = time.time() - t0

    t0 = time.time()
    for p in range(35):
        x = load_raw(work / f'patch_p{p:04d}.raw', (1, 3, 384, 384))
        o = models.run('pe', {'patch_1': x})
        d = work / 'out_pe' / f'Result_{p}'
        d.mkdir(parents=True, exist_ok=True)
        for k in PE_OUTS:
            o[k].astype(np.float32).tofile(d / f'{k}.raw')
    t['pe'] = time.time() - t0

    t0 = time.time()
    o = models.run('ie', {'image_1': load_raw(work / 'x2.raw', (1, 3, 384, 384))})
    d = work / 'out_ie' / 'Result_0'
    d.mkdir(parents=True, exist_ok=True)
    o['image_features'].astype(np.float32).tofile(d / 'image_features.raw')
    t['ie'] = time.time() - t0

    t0 = time.time(); host('merge', work, work / 'out_pe', work / 'out_ie'); t['merge'] = time.time() - t0

    t0 = time.time()
    df = np.array([fpx / ow], dtype=np.float32)
    df.tofile(work / 'disparity_factor.raw')
    rest = rest_chain(models, work, df)
    rest['disparity'].astype(np.float32).tofile(work / 'disparity.raw')
    rest['delta'].astype(np.float32).tofile(work / 'delta.raw')
    t['rest'] = time.time() - t0

    t0 = time.time(); host('post', work, fpx, ow, oh, work / 'output.ply'); t['post'] = time.time() - t0
    return t


def rest_chain(models, src_dir, disparity_factor):
    """rest_a -> rest_b -> rest_c on the merged features found in src_dir."""
    src_dir = Path(src_dir)
    pool = {
        'image': load_raw(src_dir / 'image.raw', (1, 3, 1536, 1536)),
        'disparity_factor': disparity_factor,
        'x_latent0': load_raw(src_dir / 'x_latent0.raw', (1, 1024, 96, 96)),
        'x_latent1': load_raw(src_dir / 'x_latent1.raw', (1, 1024, 96, 96)),
        'x0_feat': load_raw(src_dir / 'x0_feat.raw', (1, 1024, 96, 96)),
        'x1_feat': load_raw(src_dir / 'x1_feat.raw', (1, 1024, 48, 48)),
        'x2_feat': load_raw(src_dir / 'x2_feat.raw', (1, 1024, 24, 24)),
        'x_lowres_feat': load_raw(src_dir / 'x_lowres_feat.raw', (1, 1024, 24, 24)),
    }
    for key in ('rest_a', 'rest_b', 'rest_c'):
        pool.update(models.run(key, pool))
    models.cache.clear()
    return {'disparity': pool['disparity'], 'delta': pool['delta']}


# ---------------------------------------------------------------- main checks

def compare_dump(dump, models, min_cos, keep):
    dump = Path(dump)
    meta = json.loads((dump / 'meta.json').read_text())
    fpx, ow, oh = meta['fpx'], meta['orig_w'], meta['orig_h']
    print(f"device dump: {dump}\n  backend={meta.get('backend')} threads={meta.get('threads')} "
          f"device={meta.get('device')} soc={meta.get('soc')} fpx={fpx:.2f} orig={ow}x{oh} "
          f"peak_rss_mb={meta.get('peak_rss_mb')}")
    if meta.get('fallback_reason'):
        print(f"  HTP not used: {meta['fallback_reason']}")

    print("\n== per-model (device inputs -> desktop FP32 model) ==")
    for p in meta.get('pe_patches', DUMP_PATCHES):
        x = load_raw(dump / f'patch_p{p:04d}.raw', (1, 3, 384, 384))
        o = models.run('pe', {'patch_1': x})
        for k in PE_OUTS:
            f = dump / 'out_pe' / f'Result_{p}' / f'{k}.raw'
            if f.exists():
                report('PE', f'patch{p}.{k}', o[k], load_raw(f), min_cos)
    o = models.run('ie', {'image_1': load_raw(dump / 'x2.raw', (1, 3, 384, 384))})
    report('IE', 'image_features', o['image_features'],
           load_raw(dump / 'out_ie' / 'Result_0' / 'image_features.raw'), min_cos)

    df = load_raw(dump / 'disparity_factor.raw')
    rest = rest_chain(models, dump, df)
    report('REST', 'disparity (from device features)', rest['disparity'], load_raw(dump / 'disparity.raw'), min_cos)
    report('REST', 'delta (from device features)', rest['delta'], load_raw(dump / 'delta.raw'), min_cos)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for n in ('image.raw', 'disparity.raw', 'delta.raw'):
            shutil.copy(dump / n, td / n)
        host('post', td, fpx, ow, oh, td / 'post.ply')
        compare_ply('POST', td / 'post.ply', dump / 'output.ply', 0.999999)

    print("\n== end-to-end (device image.raw -> full desktop pipeline) ==")
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix='sharp_ref_'))
    t = run_desktop(models, dump / 'image.raw', fpx, ow, oh, work)
    print("  desktop timings (s): " + " ".join(f"{k}={v:.1f}" for k, v in t.items()))
    report('E2E', 'patch_p0000 (C pre)', load_raw(work / 'patch_p0000.raw'), load_raw(dump / 'patch_p0000.raw'), 0.999999)
    for n in MERGED:
        report('E2E', n, load_raw(work / f'{n}.raw'), load_raw(dump / f'{n}.raw'), min_cos)
    report('E2E', 'disparity', load_raw(work / 'disparity.raw'), load_raw(dump / 'disparity.raw'), min_cos)
    report('E2E', 'delta', load_raw(work / 'delta.raw'), load_raw(dump / 'delta.raw'), min_cos)
    compare_ply('E2E', work / 'output.ply', dump / 'output.ply', min_cos)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)


def make_reference(image, models, out):
    """Desktop-only run on an image, written as a dump-like directory (+ meta.json)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    line = host('prep', image, out / 'image.raw').strip().splitlines()[-1]
    fpx, ow, oh = float(line.split()[0]), int(line.split()[1]), int(line.split()[2])
    t = run_desktop(models, out / 'image.raw', fpx, ow, oh, out)
    meta = {'backend': 'desktop-onnxruntime-fp32', 'threads': models.threads, 'fpx': fpx,
            'orig_w': ow, 'orig_h': oh, 'device': 'desktop', 'soc': '', 'pe_patches': DUMP_PATCHES,
            'desktop_s': t}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"reference written to {out}: fpx={fpx:.2f} orig={ow}x{oh} timings={t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', help='pulled device dump dir (contains meta.json)')
    ap.add_argument('--onnx_dir', default=str(ROOT / 'output' / 'onnx'))
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--quantized', action='store_true',
                    help='device ran quantised models (HTP W8A16): relax cos threshold to 0.98')
    ap.add_argument('--min_cos', type=float, default=None)
    ap.add_argument('--keep', default=None, help='keep the desktop end-to-end work dir here')
    ap.add_argument('--reference_image', help='desktop-only: run the full pipeline on this image')
    ap.add_argument('--out', help='output dir for --reference_image')
    a = ap.parse_args()

    models = Models(a.onnx_dir, a.threads)
    if a.reference_image:
        make_reference(a.reference_image, models, a.out or 'ref_run')
        return
    if not a.dump:
        ap.error('--dump or --reference_image required')
    min_cos = a.min_cos if a.min_cos is not None else (0.98 if a.quantized else 0.999)
    compare_dump(a.dump, models, min_cos, a.keep)

    failed = [(g, n) for g, n, _, ok in RESULTS if not ok]
    print(f"\nRESULT: {'PASS' if not failed else 'FAIL'} ({len(RESULTS) - len(failed)}/{len(RESULTS)} checks ok, "
          f"min_cos={min_cos})")
    for g, n in failed:
        print(f"  failed: [{g}] {n}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
