"""C1 task 5 -- census of the frozen rule's verdict on a DEFINED population.

POPULATION.  timm 1.0.27, `timm.list_models(pretrained=True)` (1703 tags),
deduplicated to architecture names (text before the first '.').  Primary
population P1: architectures whose DEFAULT pretrained tag
(`timm.models.get_pretrained_cfg(arch)`, i.e. the first listed tag) has
num_classes == 1000 and 'in1k' in the tag name.  Secondary population P2:
architectures with ANY pretrained tag satisfying the same condition (reported
for sensitivity; the structural verdict does not depend on the weights).

MEASUREMENT.  Structure only.  `timm.create_model(arch, pretrained=False)`
(random weights; the verdict is read from module classes), eval mode, one
forward_features on a random (1, 3, H, W) input at the default tag's input size,
then `frozen_rule.structural(m, f)` (read-only import) and, for comparison,
`rule_v2.structural_v2` (v2b = operator-trace repair).  Models with more than
1e9 parameters (counted on the meta device) are skipped.  Per-model wall-clock
limit enforced by the driver (killed worker -> 'timeout').

usage:  c5_census.py                 driver (spawns workers)
        c5_census.py --worker K N    worker K of N (internal)
"""
import os, sys, json, time, subprocess, traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
WD = os.path.join(OUT, 'c5_work')
os.makedirs(WD, exist_ok=True)
TIMEOUT = 240.0
NW = 6
MAXP = 1e9


def population():
    import timm
    from timm.models import get_pretrained_cfg
    tags = timm.list_models(pretrained=True)
    archs = sorted(set(t.split('.')[0] for t in tags))
    p1, p2 = [], []
    for a in archs:
        c = get_pretrained_cfg(a)
        if c is not None and c.num_classes == 1000 and 'in1k' in (c.tag or ''):
            p1.append(a)
        anyin1k = False
        for t in tags:
            if t.split('.')[0] != a:
                continue
            try:
                ct = get_pretrained_cfg(t)
            except Exception:
                ct = None
            if ct is not None and ct.num_classes == 1000 and 'in1k' in t.split('.', 1)[1]:
                anyin1k = True
                break
        if anyin1k:
            p2.append(a)
    return tags, archs, p1, p2


def worker(k, n):
    import torch, timm
    from timm.models import get_pretrained_cfg
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))
    import frozen_rule as FR
    import rule_v2 as R2
    todo = json.load(open(os.path.join(WD, 'todo.json')))
    mine = [a for i, a in enumerate(todo) if i % n == k]
    outp = os.path.join(WD, 'res_%d.jsonl' % k)
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            try:
                done.add(json.loads(line)['arch'])
            except Exception:
                pass
    stp = os.path.join(WD, 'start_%d.txt' % k)
    for a in mine:
        if a in done:
            continue
        open(stp, 'w').write('%s\t%f' % (a, time.time()))
        rec = dict(arch=a)
        t0 = time.time()
        try:
            cfg = get_pretrained_cfg(a)
            rec['tag'] = cfg.tag
            isz = list(cfg.input_size)
            rec['input_size'] = isz
            try:
                with torch.device('meta'):
                    mm = timm.create_model(a, pretrained=False)
            except Exception:
                rec['meta_failed'] = True       # some inits copy data; count on CPU instead
                mm = timm.create_model(a, pretrained=False)
            npar = sum(p.numel() for p in mm.parameters())
            del mm
            rec['n_params'] = int(npar)
            if npar > MAXP:
                rec['skipped'] = 'params>1e9'
            else:
                m = timm.create_model(a, pretrained=False).eval()
                dev = 'cuda'
                try:
                    m = m.to(dev)
                    x = torch.randn(1, *isz, device=dev)
                    with torch.no_grad():
                        f = m.forward_features(x)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    dev = 'cpu'; m = m.cpu(); x = torch.randn(1, *isz)
                    with torch.no_grad():
                        f = m.forward_features(x)
                rec['device'] = dev
                if not torch.is_tensor(f):
                    rec['feat_type'] = type(f).__name__
                st = R2.structural_v2(m, f)        # runs frozen_rule.structural first
                rec.update(verdict=st['verdict'], agg=st['agg'], agg_name=st['agg_name'],
                           blocker=st['blocker'], post=[list(p) for p in st['post']],
                           v2a=st['v2a'], v2b=st['v2b'], module_pool=st['module_pool'],
                           aten_classes=st['aten_classes'])
                del m, f, x
                if dev == 'cuda':
                    torch.cuda.empty_cache()
        except Exception as e:
            rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
        rec['secs'] = time.time() - t0
        with open(outp, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    open(stp, 'w').write('DONE')


def driver(retry=False):
    import glob
    if retry:
        # second pass over architectures whose last record is an error (not timeouts)
        last = {}
        for f in sorted(glob.glob(os.path.join(WD, 'res_*.jsonl'))):
            for line in open(f):
                d = json.loads(line)
                last[d['arch']] = d
        todo = sorted(a for a, d in last.items() if 'error' in d and not d['error'].startswith('timeout'))
        for f in glob.glob(os.path.join(WD, 'res_*.jsonl')):
            os.replace(f, f.replace('res_', 'pass1_res_'))
        print('retry %d' % len(todo), flush=True)
    else:
        tags, archs, p1, p2 = population()
        todo = sorted(set(p1) | set(p2))
        json.dump(dict(n_tags=len(tags), n_archs=len(archs), p1=p1, p2=p2),
                  open(os.path.join(OUT, 'c5_population.json'), 'w'), indent=1)
        print('tags %d archs %d P1 %d P2 %d todo %d' % (len(tags), len(archs), len(p1), len(p2), len(todo)),
              flush=True)
    json.dump(todo, open(os.path.join(WD, 'todo.json'), 'w'))
    py = sys.executable
    procs = {}

    def start(k):
        sp = os.path.join(WD, 'start_%d.txt' % k)
        if os.path.exists(sp):
            os.remove(sp)
        procs[k] = subprocess.Popen([py, os.path.abspath(__file__), '--worker', str(k), str(NW)],
                                    stdout=open(os.path.join(WD, 'w%d.log' % k), 'a'),
                                    stderr=subprocess.STDOUT)
    for k in range(NW):
        start(k)
    while procs:
        time.sleep(5)
        for k in list(procs):
            p = procs[k]
            sp = os.path.join(WD, 'start_%d.txt' % k)
            if p.poll() is not None:
                st = open(sp).read() if os.path.exists(sp) else ''
                if st == 'DONE':
                    del procs[k]
                else:
                    # crashed mid-model: record and restart
                    a = st.split('\t')[0] if st else None
                    if a:
                        with open(os.path.join(WD, 'res_%d.jsonl' % k), 'a') as fh:
                            fh.write(json.dumps(dict(arch=a, error='worker crashed')) + '\n')
                    start(k)
                continue
            if os.path.exists(sp):
                st = open(sp).read()
                if st and st != 'DONE':
                    a, t = st.split('\t')
                    if time.time() - float(t) > TIMEOUT:
                        p.kill(); p.wait()
                        with open(os.path.join(WD, 'res_%d.jsonl' % k), 'a') as fh:
                            fh.write(json.dumps(dict(arch=a, error='timeout>%ds' % TIMEOUT)) + '\n')
                        print('timeout', a, flush=True)
                        start(k)
    print('all workers done', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        driver(retry='--retry' in sys.argv)
