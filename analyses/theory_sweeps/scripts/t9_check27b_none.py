"""TASK 9 side check -- probe27b_scratch_imagenette.json reports the `none`
ordering at exactly the same top-1 (0.7268789808917198 = 2853/3925) for seeds
0, 1 and 2, while `pre` and `post` differ by seed and seeds 3-4 (run later)
differ. Re-train `none` for seeds 0 and 1 with the archived function
(scripts/probe27b_scratch.run, EPOCHS = 30 as in the archived meta) and record
whether the seed changes the result. Nothing is written outside theory_sweeps/out."""
import os, sys, json
os.environ.setdefault('EPOCHS', '30')
os.environ.setdefault('DS', 'imagenette')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUTF = os.path.join(os.path.dirname(HERE), 'out', 't9_check27b_none.json')


def main():
    import probe27b_scratch as P
    dtr, dte, ncls, img, patch = P.data()
    res = {}
    for s in (0, 1):
        a, c, r, sec = P.run('none', s, dtr, dte, ncls, img, patch)
        res[f'none_s{s}'] = dict(acc=a, corrNS=c, Rhalf=r, sec=sec)
        print(f'none seed {s}: acc={a:.6f} ({a*len(dte):.0f}/{len(dte)}) corrNS={c:.10f} ({sec:.0f}s)', flush=True)
        json.dump(res, open(OUTF, 'w'), indent=1)


if __name__ == '__main__':
    main()
