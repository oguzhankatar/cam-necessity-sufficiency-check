"""C3 tasks 5 and 6 -- read-only extraction from the archived runs.

Task 5: certificate separation 1 - corr(N,S) per (dataset, backbone) for the
  aggregator arm (edited_lse_tau1, LSE at T=+1, GAP weights reused) and the
  relocation arm (edited_lnpost, ViT only), with the number of probe pairs and
  the ADDITIVE flags of the collapsed and edited heads.  corr is the minimum
  Pearson corr over the probe pairs (w2_run.py -> wlib.certify_head).
Task 6: the design facts of every primary cell as recorded in the archived
  w2_<ds>_<bk>.json (timm id, input size, grid, channels, n_fit, n_eval,
  n_records, bank, tau, basis, code_sha, classification of the heads) and
  the dataset inventory w0_data.json.
Nothing under scripts/ or outputs/ is written.
"""
import os
import json
from localisation_reanalysis_lib import WO, OUT, DATASETS, BACKBONES, METRICS, save


def main():
    cells = []
    for ds in DATASETS:
        for bk in BACKBONES:
            j = json.load(open(os.path.join(WO, 'w2_%s_%s.json' % (ds, bk))))
            ce = j['certificate']
            e = dict(dataset=ds, backbone=bk, timm=j['timm'], input=j['input'], grid=j['grid'],
                     C=j['C'], eval_res=j['eval_res'], n_fit=j['n_fit'], n_eval=j['n_eval'],
                     n_records=j['n_records'], n_bank=j['n_bank'], bank_idx=j['bank_idx'],
                     basis=j['basis'], tau=j['tau'], n_thresholds=j['n_thresholds'],
                     deltas=j['deltas'], code_sha=j['code_sha'], metrics=METRICS[ds],
                     classification=j['classification'],
                     n_primary=len(METRICS[ds]) * (8 if bk == 'vitb16' else 4))
            for k, v in ce.items():
                e['cert_' + k] = dict(ADDITIVE=v['ADDITIVE'], maxNS=v['maxNS'], Rhalf=v['Rhalf'],
                                      I2=v['I2'], corr=v['corrNS_recorded_never_used'],
                                      sep=1 - v['corrNS_recorded_never_used'],
                                      n_pairs=v['n_pairs'], T=v['T'])
            cells.append(e)
    inv = json.load(open(os.path.join(WO, 'w0_data.json')))

    print('Task 5 -- certificate separation 1-corr(N,S) (min-corr pair of n probe pairs)')
    print('%-9s %-9s %6s %6s %8s %8s  %s' % ('dataset', 'backbone', 'n_pair', 'LSE', 'LN', 'gapADD', 'lseADD/lnADD'))
    for e in cells:
        l, g = e['cert_edited_lse_tau1'], e['cert_collapsed_gap']
        ln = e.get('cert_edited_lnpost')
        print('%-9s %-9s %6d %6.3f %8s %8s  %s/%s' % (e['dataset'], e['backbone'], l['n_pairs'], l['sep'],
              '%.3f' % ln['sep'] if ln else '--', g['ADDITIVE'], l['ADDITIVE'], ln['ADDITIVE'] if ln else '--'))
    # verdicts vs collapsed per cell and arm (from task 1), beside the separation
    from collections import Counter
    from scipy.stats import spearmanr
    rows = json.load(open(os.path.join(OUT, 't1_t2_contrasts.json')))['rows_vs_collapsed']
    sep_vs_wins = []
    print('\nseparation beside verdicts vs collapsed (win/loss/tie)')
    for e in cells:
        for arm, maps, key in (('aggregator', ('N_lse', 'S_lse'), 'cert_edited_lse_tau1'),
                               ('relocation', ('N_ln', 'S_ln'), 'cert_edited_lnpost')):
            if key not in e:
                continue
            s = [r for r in rows if r['dataset'] == e['dataset'] and r['backbone'] == e['backbone'] and r['a'] in maps]
            c = Counter(r['sig'] for r in s)
            d = dict(dataset=e['dataset'], backbone=e['backbone'], arm=arm, sep=e[key]['sep'], n=len(s),
                     win=c['win'], loss=c['loss'], tie=c['tie'])
            sep_vs_wins.append(d)
            print('  %-9s %-9s %-10s sep %.3f  n=%d win %d loss %d tie %d'
                  % (d['dataset'], d['backbone'], arm, d['sep'], d['n'], d['win'], d['loss'], d['tie']))
    ag = [d for d in sep_vs_wins if d['arm'] == 'aggregator']
    rho = spearmanr([d['sep'] for d in ag], [(d['win'] - d['loss']) / d['n'] for d in ag])
    print('  aggregator, 12 cells: Spearman(sep, (win-loss)/n) = %.3f (p=%.3f)' % (rho[0], rho[1]))
    print('\nTask 6 -- design facts')
    for e in cells:
        print('%-9s %-9s %-45s in=%d grid=%s C=%d fit=%d eval=%d rec=%d bank=%d tau=%g sha=%s primary=%d cls=%s'
              % (e['dataset'], e['backbone'], e['timm'], e['input'], e['grid'], e['C'], e['n_fit'],
                 e['n_eval'], e['n_records'], e['n_bank'], e['tau'], e['code_sha'], e['n_primary'],
                 {k: round(list(v.values())[0], 4) for k, v in e['classification'].items()}))
    print('total primary contrasts:', sum(e['n_primary'] for e in cells))
    save('t5_t6_archive.json', dict(cells=cells, inventory=inv, sep_vs_verdicts=sep_vs_wins,
                                    spearman_aggregator=dict(rho=rho[0], p=rho[1], n=len(ag))))


if __name__ == '__main__':
    main()
