"""K1 / K1b per-model runner: original, audited and conservative checks, and the float64
certificate, on one list of models. Random weights throughout (seed 0).

usage: python k1_run.py <list> [--cpu] [--threads N]
  <list>: timm   -> the 148 development + held-out architectures, then the census p1
                    entries (timm 1.0.27, 895 classified), each name once
          tv     -> every torchvision 0.21 classifier (torchvision.models.list_models)
Appends one JSON line per model to ../out/k1_run_<list>.jsonl (resumable).

Per model
  checks       on a 2-image N(0,1) batch, float32, the model's default input size:
               check_model (original = verdict_check, audited = verdict), the conservative
               check (K1) and the post-hoc operator-guard variant; wall-clock per check.
  certificate  float64 (model and input), 6 N(0,1) images = 3 pairs, the second image of
               each pair the reference: eps1-eps3 by the literal per-cell loop at the head
               (analyses/check_audit/scripts/check_audit_lib.measure, the article's certificate), with the
               post-aggregator normalisations deleted for the normalisation-only test; once
               on the model's own features and once on a unit-scale N(0,1) feature map of the
               same shape (as analyses/check_audit/scripts/a1_audit.py). Wall-clock of the own-feature run
               (forward_features + certificate) is the certificate cost.
Device: CPU for everything (the GPU is shared; on it each of the thousands of tiny head
calls of the certificate waits behind other jobs). Work is split over processes by
--shard i --nshard k; models above 300 M parameters (census record) form their own shard
(--big) so that at most one large float64 model is resident at a time.
"""
import os, sys, json, time, traceback, argparse, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k1lib as K
sys.path.insert(0, os.path.join(K.ROOT, 'analyses', 'check_audit', 'scripts'))
import torch
import timm
import check_audit_lib as L
from check_model import check_model

ap = argparse.ArgumentParser()
ap.add_argument('list')
ap.add_argument('--gpu', action='store_true')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshard', type=int, default=1)
ap.add_argument('--big', action='store_true')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--only', default=None)
ap.add_argument('--retry', action='store_true',
                help='re-run the entries that ended in an error; the certificate then uses a freshly built '
                     'model converted to float64 before its first forward (models that cache float32 '
                     'buffers on their first forward, e.g. ConViT)')
ap.add_argument('--body32', action='store_true',
                help='with --retry: run the trunk of the certificate in float32 (head and residuals float64)')
ap.add_argument('--reinit', action='store_true',
                help='supplementary: re-certify the entries whose readout is identically zero at '
                     'initialisation (absN_max = 0), after drawing every all-zero weight tensor '
                     '(dim >= 2) from N(0, 0.02^2); output k1_reinit_<list>.jsonl')
a = ap.parse_args()
torch.set_num_threads(a.threads)
warnings.filterwarnings('ignore')
OUT = os.path.join(HERE, '..', 'out')
path = os.path.join(OUT, 'k1_run_%s_%s.jsonl' % (a.list, ('retry32' if a.body32 else 'retry') if a.retry else 'big' if a.big else '%d' % a.shard))
if a.reinit:
    path = os.path.join(OUT, 'k1_reinit_%s.jsonl' % a.list)
GPU = torch.cuda.is_available() and a.gpu
if GPU:
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(5.5e9 / tot)

if a.list == 'timm':
    names = []
    for s in 'DAB':
        names += [(n, 'arch148_' + s) for n in open(os.path.join(K.ROOT, 'scripts', 'heldout', 'names_%s.txt' % s)).read().split()]
    C1 = os.path.join(K.ROOT, 'analyses', 'check_census', 'out')
    R = json.load(open(os.path.join(C1, 'c5_census.json')))['records']
    pop = json.load(open(os.path.join(C1, 'c5_population.json')))
    census = [n for n in pop['p1'] if n in R and 'verdict' in R[n]]
    assert len(census) == 895
    seen = {n for n, _ in names}
    names += [(n, 'census') for n in census if n not in seen]
    insize = {n: tuple(R[n]['input_size']) for n in census}
    npar = {n: R[n].get('n_params', 0) for n in census}
else:
    import k1_tv as TVM
    names = [(n, 'torchvision') for n in TVM.tv_names()]
    insize = {}
    npar = {'vit_h_14': 632e6, 'regnet_y_128gf': 645e6}
big = [x for x in names if npar.get(x[0], 0) > 300e6]
small = [x for x in names if npar.get(x[0], 0) <= 300e6]
names = big if a.big else small[a.shard::a.nshard]
if a.retry:
    import glob
    err = set()
    for p_ in sorted(glob.glob(os.path.join(OUT, 'k1_run_%s_*.jsonl' % a.list))):
        if p_.endswith('_retry32.jsonl'):
            continue
        for l in open(p_):
            r_ = json.loads(l)
            if 'error' in r_:
                err.add(r_['name'])
            elif p_.endswith('_retry.jsonl'):
                err.discard(r_['name'])
    names = [x for x in big + small if x[0] in err]
if a.reinit:
    import glob
    zero = {}
    for p_ in sorted(glob.glob(os.path.join(OUT, 'k1_run_%s_*.jsonl' % a.list))):
        for l in open(p_):
            r_ = json.loads(l)
            if 'shipped' in r_:
                zero[r_['name']] = r_['shipped']['absN_max'] == 0.0
    names = [x for x in big + small if zero.get(x[0])]
if a.only:
    keep = set(a.only.split(','))
    names = [x for x in names if x[0] in keep]
done = set()
if os.path.exists(path):
    for l in open(path):
        try:
            done.add(json.loads(l)['name'])
        except Exception:
            pass


def build(n):
    torch.manual_seed(0)
    if a.list == 'tv':
        m = TVM.TV(n)
        return m, tuple(m.pretrained_cfg['input_size'])
    m = timm.create_model(n, pretrained=False)
    size = insize.get(n) or tuple(m.pretrained_cfg.get('input_size', (3, 224, 224)))
    return m, size


def sync():
    if GPU:
        torch.cuda.synchronize()


class PadCells(L.Cells):
    """list feature with token sequences of unequal width (CrossViT): each sequence's cells are
    zero-padded to the widest channel count in the cell matrix and sliced back for the head.
    The padding is identical in both images of a pair, so substitution is unaffected."""

    def __init__(s, f):
        import numpy as np
        s.islist, s.fmt = True, 'LIST-PAD'
        s.lens = [t.shape[1] for t in f]
        s.Cs = [t.shape[2] for t in f]
        s.C = max(s.Cs)
        s.prefix = [int(i) for i in np.cumsum([0] + s.lens[:-1])]
        s.T = int(sum(s.lens))

    def cells(s, f):
        return torch.cat([torch.nn.functional.pad(t, (0, s.C - t.shape[2])) for t in f], dim=1)

    def feat(s, X):
        return [t[..., :c] for t, c in zip(torch.split(X, s.lens, dim=1), s.Cs)]


def cells_of(m, f):
    if isinstance(f, (list, tuple)) and len({t.shape[2] for t in f}) > 1:
        return PadCells(f)
    return L.cells_for_timm(m, f)


def certificate(m, size, dev, norms_from_tail):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, *size, generator=g, dtype=torch.float64).to(dev)
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        if a.body32:
            # the trunk runs in float32 on a float32 copy (models that force float32 internally, e.g. the
            # mixed-RoPE ViTs); its features are cast to float64 and the head and every residual are float64
            m32 = build(m.k1_name)[0].eval().float()
            f = m32.forward_features(x.float())
            f = [t.double() for t in f] if isinstance(f, (list, tuple)) else f.double()
            del m32
        else:
            f = m.forward_features(x)
    mods = dict(m.named_modules())
    norms = [p for p in norms_from_tail if L.is_norm_name(mods[p])]
    cv = cells_of(m, f)
    ship, nonorm, lab = L.measure(m, f, cv, norms)
    sync(); secs = time.perf_counter() - t0
    with torch.no_grad():
        z0 = m.forward_head(f); z1 = m.forward_head(cv.feat(cv.cells(f)))
    rt = float((z0 - z1).abs().max() / (z0.abs().max() + 1e-300))
    gu = torch.Generator().manual_seed(1)
    fu = ([torch.randn(t.shape, generator=gu, dtype=t.dtype).to(dev) for t in f]
          if isinstance(f, (list, tuple)) else torch.randn(f.shape, generator=gu, dtype=f.dtype).to(dev))
    ship_u, nonorm_u, lab_u = L.measure(m, fu, cv, norms)
    fstd = float(torch.cat([t.flatten() for t in (f if isinstance(f, (list, tuple)) else [f])]).std())
    return dict(label_own=lab, label_unit=lab_u, label=lab if lab == lab_u else lab_u,
                shipped=ship, nonorm=nonorm, shipped_unit=ship_u, nonorm_unit=nonorm_u,
                norms_deleted=norms, fmt=cv.fmt, T=cv.T, C=cv.C, n_prefix=len(cv.prefix),
                feat_std=fstd, head_roundtrip=rt, secs_certificate=secs)


fo = open(path, 'a')
t00 = time.time()
for n, src in names:
    if n in done:
        continue
    rec = dict(name=n, set=src)
    t0 = time.time()
    m = None
    try:
        m, size = build(n)
        m.eval()
        rec['n_params'] = int(sum(p.numel() for p in m.parameters()))
        rec['input_size'] = list(size)
        dev = 'cuda' if (GPU and rec['n_params'] <= 300e6) else 'cpu'
        for attempt in range(2):
            try:
                m = m.float().to(dev)
                g = torch.Generator().manual_seed(0)
                x2 = torch.randn(2, *size, generator=g).to(dev)
                sync(); t1 = time.perf_counter()
                cm = check_model(m, x=x2, warn=False)
                sync(); rec['secs_original'] = time.perf_counter() - t1
                s = K.check_model_sound(m, x2, return_timing=True)
                sync(); rec['secs_sound'] = s.pop('secs')
                gg = K.check_model_sound(m, x2, op_guard=True)
                rec.update(device=dev, original=cm['verdict_check'], audited=cm['verdict'],
                           rules_fired=cm['rules_fired'], aggregator=cm['aggregator'], tail=cm['tail'],
                           sound=s['verdict'], sound_failed=s['failed'], sound_why=s['why'],
                           sound_branch=s['branch'], operator_verdict=s['operator_verdict'],
                           posthoc_opguard=gg['verdict'], posthoc_failed=gg['failed'], posthoc_why=gg['why'])
                if a.retry:
                    del m
                    m, _ = build(n)
                    m = m.eval().to(dev)
                    rec['certificate_fresh_model'] = True
                if a.reinit:
                    g0 = torch.Generator().manual_seed(0)
                    rz = []
                    with torch.no_grad():
                        for pn, pp in m.named_parameters():
                            if pp.dim() >= 2 and not bool(pp.any()):
                                pp.copy_(torch.randn(pp.shape, generator=g0) * 0.02)
                                rz.append(pn)
                    rec['reinitialised'] = rz
                m = m.double()
                m.k1_name = n
                if a.body32:
                    rec['certificate_trunk_float32'] = True
                rec.update(certificate(m, size, dev, [p[0] for p in cm['tail']]))
                break
            except torch.cuda.OutOfMemoryError:
                if dev == 'cpu':
                    raise
                rec['gpu_oom'] = True
                dev = 'cpu'
                m = m.cpu()
                torch.cuda.empty_cache()
        c = rec['shipped']
        print('%-40s %-8s orig %-22s aud %-22s sound %-22s | meas %-12s eps1 %.1e eps2 %.1e (%s) %.0fs'
              % (n, dev, rec['original'], rec['audited'], rec['sound'], rec['label'], c['eps1'], c['eps2'],
                 rec['label_own'] + ('' if rec['label_own'] == rec['label_unit'] else '/unit ' + rec['label_unit']),
                 time.time() - t0), flush=True)
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
        rec['tb'] = traceback.format_exc()[-1500:]
        print('%-40s ERROR %s' % (n, rec['error'][:160]), flush=True)
    rec['secs_total'] = time.time() - t0
    fo.write(json.dumps(rec, default=float) + '\n'); fo.flush()
    del m
    if GPU:
        torch.cuda.empty_cache()
print('total %.0f s' % (time.time() - t00))
