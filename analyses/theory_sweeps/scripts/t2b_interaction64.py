"""TASK 2 complement -- I2, Rhalf and corrNS for every probe-1 head (plus the
post-mean LayerNorm head ln_post) in FLOAT64, with the archived estimator
itself: scripts/probe1_aggregator.interaction, imported read-only, the same 48
images x 3 bank references and the same RNG stream (RandomState(1)) as probe 1.
The archived probe1_aggregator.json values were computed in float32."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L
from probe1_aggregator import interaction  # noqa: E402  (scripts/ on sys.path via theory_sweeps_lib)

HEADS = ['gap_fc', 'gap_mlp', 'ln_gap', 'tok_mlp', 'ln_post', 'lse0.25', 'lse0.5', 'lse1.0',
         'lse2.0', 'lse4.0', 'lse8.0', 'maxpool', 'attn', 'bilin']


def main():
    a1 = json.load(open(os.path.join(L.OUTP, 'probe1_aggregator.json')))
    imgs, bank_idx, _ = L.probe1_sample()
    X = L.batch('val', imgs).double(); Bk = L.batch('val', bank_idx).double()
    res = {}
    for nm in HEADS:
        h, acc = L.get_head(nm, double=True)
        rs2 = np.random.RandomState(1)
        I2s, Rhs, cs = [], [], []
        with torch.no_grad():
            for k in range(X.shape[0]):
                A = X[k]; c = int(h(A[None]).argmax(1))
                for b in range(3):
                    B = Bk[rs2.randint(Bk.shape[0])]
                    i2, rh, cc, _ = interaction(h, A, B, c, rs=rs2)
                    I2s.append(i2); Rhs.append(rh)
                    if not np.isnan(cc): cs.append(cc)
        ar = a1.get(nm, {})
        res[nm] = dict(acc=acc, I2=float(np.mean(I2s)), Rhalf=float(np.mean(Rhs)),
                       corrNS=float(np.mean(cs)), one_minus_corr=float(1 - np.mean(cs)),
                       n_pairs=len(I2s), n_corr=len(cs),
                       archived_float32=dict(I2=ar.get('I2'), Rhalf=ar.get('Rhalf'), corrNS=ar.get('corrNS')))
        r = res[nm]
        print(f"{nm:8s} acc={acc:.4f} 1-corr={r['one_minus_corr']:.4g} Rhalf={r['Rhalf']:.4g} I2={r['I2']:.4g} | "
              f"f32: Rhalf={ar.get('Rhalf')} I2={ar.get('I2')}", flush=True)
        L.save('t2b_interaction64.json', res)


if __name__ == '__main__':
    main()
