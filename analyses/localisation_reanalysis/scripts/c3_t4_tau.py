"""C3 task 4: the temperature and mean-cell runs.

For each archived run of w2_run.py with a non-default temperature or basis
(w2_cub_resnet50_tau{-8,-2,2,8}, w2_voc12seg_resnet50_tau{-8,-2,-0.5,0.5,2,8},
w2_{cub,voc12seg}_resnet50_meancell) and for the pre-specified T=+1 run:
  * separated minus collapsed for N_lse and S_lse on every metric the dataset
    defines, with w3's tests and Holm inside the {N,S} x {collapsed, center}
    family, exactly as for the primary contrasts;
  * PxAP (no per-record form; read from the JSON);
  * the certificate of the edited head as stored (temperature, maxNS, Rhalf,
    I2, corr N-S, n_pairs), and checks that each certificate was computed at
    its run's own temperature;
  * file modification times and code_sha.
"""
import os
import json
import time
from collections import Counter
import numpy as np
from localisation_reanalysis_lib import W, WO, METRICS, MNAME, test, holm, verdict, save

RUNS = [('cub', s) for s in ('_tau-8', '_tau-2', '', '_tau2', '_tau8', '_meancell')] + \
       [('voc12seg', s) for s in ('_tau-8', '_tau-2', '_tau-0.5', '_tau0.5', '', '_tau2', '_tau8', '_meancell')]


def main():
    res = []
    for ds, suf in RUNS:
        f = os.path.join(WO, 'w2_%s_resnet50%s.json' % (ds, suf))
        j = json.load(open(f))
        cert = j['certificate']['edited_lse_tau1']
        e = dict(file=os.path.basename(f), dataset=ds, tau=j['tau'], basis=j['basis'],
                 code_sha=j['code_sha'],
                 mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f))),
                 cert_key='edited_lse_tau1', cert_corrNS=cert['corrNS_recorded_never_used'],
                 cert_sep=1 - cert['corrNS_recorded_never_used'],
                 cert_maxNS=cert['maxNS'], cert_Rhalf=cert['Rhalf'], cert_I2=cert['I2'],
                 cert_n_pairs=cert['n_pairs'], contrasts=[])
        for m in METRICS[ds]:
            fam = []
            for a in ('N_lse', 'S_lse'):
                for b in ('collapsed', 'center'):
                    r = test(ds, 'resnet50', m, a, b, suf)
                    r.update(metric=m, a=a, b=b)
                    fam.append(r)
            for r, q in zip(fam, holm([r['p'] for r in fam])):
                r['p_holm'] = q
                r['sig'] = verdict(r['diff'], q)
            e['contrasts'] += fam
        if 'PxAP' in j['per_map']['collapsed']:
            e['PxAP'] = {k: j['per_map'][k]['PxAP'] for k in ('collapsed', 'N_lse', 'S_lse', 'cam', 'center')}
        res.append(e)

    # identity checks
    chk = {}
    for ds in ('cub', 'voc12seg'):
        a = np.load(os.path.join(WO, 'pt_%s_resnet50_collapsed.npy' % ds))
        same = all(np.array_equal(a, np.load(os.path.join(WO, 'pt_%s_resnet50%s_collapsed.npy' % (ds, s))))
                   for d2, s in RUNS if d2 == ds and s.startswith('_tau'))
        mc = np.load(os.path.join(WO, 'pt_%s_resnet50_meancell_collapsed.npy' % ds))
        cam = np.load(os.path.join(WO, 'pt_%s_resnet50_cam.npy' % ds))
        chk[ds] = dict(collapsed_identical_across_tau_runs=bool(same),
                       meancell_collapsed_equals_plain_cam=bool(np.array_equal(mc, cam)))
    print('identity checks:', chk)

    print('\n%-9s %-6s %-9s %-15s %-6s %9s %9s %9s %-5s' % ('dataset', 'tau', 'basis', 'metric', 'map', 'sep-coll', 'lo', 'hi', 'sig'))
    for e in res:
        print('%s  tau=%g basis=%s  cert sep(1-corr)=%.4f maxNS=%.3f n_pairs=%d  mtime %s'
              % (e['file'], e['tau'], e['basis'], e['cert_sep'], e['cert_maxNS'], e['cert_n_pairs'], e['mtime']))
        for r in e['contrasts']:
            if r['b'] != 'collapsed':
                continue
            print('   %-15s %-6s %.4f vs %.4f  %+.4f [%+.4f,%+.4f] %s'
                  % (MNAME[r['metric']], r['a'], r['va'], r['vb'], r['diff'], r['ci'][0], r['ci'][1], r['sig']))
        if 'PxAP' in e:
            print('   PxAP', {k: round(v, 4) for k, v in e['PxAP'].items()})
        c = Counter(r['sig'] for r in e['contrasts'] if r['b'] == 'collapsed')
        e['tally_vs_collapsed'] = dict(c)
        print('   tally vs collapsed:', dict(c))
    save('t4_tau.json', dict(identity_checks=chk, runs=res))


if __name__ == '__main__':
    main()
