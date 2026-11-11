"""Fit every head the C4 analyses need, once, with the probe-1 recipe, and
compare top-1 with the archived probe1_aggregator.json / probe7_andor.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L

P1 = ['gap_fc', 'gap_mlp', 'ln_gap', 'tok_mlp', 'lse0.25', 'lse0.5', 'lse1.0',
      'lse2.0', 'lse4.0', 'lse8.0', 'maxpool', 'attn', 'bilin']
EXTRA = ['ln_post', 'lse-8.0', 'lse-4.0', 'lse-2.0', 'lse-1.0', 'lse-0.5', 'lse0.0']


def main():
    a1 = json.load(open(os.path.join(L.OUTP, 'probe1_aggregator.json')))
    a7 = json.load(open(os.path.join(L.OUTP, 'probe7_andor.json')))
    rep = {}
    for nm in P1 + EXTRA:
        t0 = time.time()
        _, acc = L.get_head(nm)
        old = a1.get(nm, a7.get(nm, {})).get('acc')
        rep[nm] = dict(acc=acc, archived_acc=old)
        print(f'{nm:9s} acc={acc:.4f} archived={old}  ({time.time()-t0:.0f}s)', flush=True)
    L.save('t0_heads.json', rep)


if __name__ == '__main__':
    main()
