"""TASKS 5, 7, 8 -- archived-JSON analyses (no model is run).

5: Student-t intervals for n = 5 seed-level results
   - probe22b: learnable aggregator at the smallest alpha learning rate, paired
     accuracy change against alpha frozen at 0 (same seed)
   - probe27b: pre - none, ViT trained from scratch, five paired seeds
7: float32 vs float64 inventory of the values the article prints as
   exact, with the float32-level residual and the float64 re-certification
8: probe30 (128 images) vs probe29 (224 images) image sets and references.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L

O = L.OUTP


def J(p):
    return json.load(open(os.path.join(O, p)))


def t5():
    d = J('probe22b_learnalpha_resnet50_imagenette.json')
    f = np.array(d['frozen0']['per_seed'])
    res = {}
    for alr in ('0.001', '0.01', '0.1'):
        a = np.array(d[f'alr{alr}_init0.0']['per_seed_acc'])
        dd = 100 * (a - f)
        m, lo, hi, se, n = L.tci(dd)
        res[f'probe22b_alr{alr}'] = dict(per_seed_pp=dd.tolist(), mean=m, t95=[lo, hi], se=se, n=n,
                                         range=[float(dd.min()), float(dd.max())],
                                         archived_percentile_boot=[100 * x for x in d[f'alr{alr}_init0.0']['dacc_ci']],
                                         alpha_per_seed=d[f'alr{alr}_init0.0']['per_seed_alpha'])
    allv = np.concatenate([np.array(res[f'probe22b_alr{a}']['per_seed_pp']) for a in ('0.001', '0.01', '0.1')])
    res['probe22b_all15'] = dict(mean=float(allv.mean()), n_above_zero=int((allv > 0).sum()), n=len(allv))
    e = J('probe27b_scratch_imagenette_5seed.json')
    for key in ('pre_minus_none', 'post_minus_none'):
        v = np.array(e[key]['per_seed'])
        m, lo, hi, se, n = L.tci(v)
        res[f'probe27b_{key}'] = dict(per_seed_pp=v.tolist(), mean=m, t95=[lo, hi], se=se, n=n,
                                      range=[float(v.min()), float(v.max())],
                                      archived_percentile_boot=e[key]['ci'])
    res['probe27b_accs'] = {k: e[k]['accs'] for k in ('none', 'pre', 'post')}
    return res


def t7():
    inv = {}
    # (i) Table 2 edited column: probe25, float32 estimator
    rows = {}
    for f in sorted(os.listdir(O)):
        if f.startswith('probe25_lnplace_') and f.endswith('.json'):
            d = J(f); sh = d['meta']['shipped']
            ed = 'pre' if sh == 'post' else 'post'
            rows[d['meta']['model']] = dict(shipped=sh, edited=ed, n_img=d['meta']['n_img'],
                                            corr_edited=d[ed]['corrNS'], one_minus_corr=1 - d[ed]['corrNS'],
                                            Rhalf=d[ed]['Rhalf'], I2=d[ed]['I2'],
                                            corr_none=d['none']['corrNS'], Rhalf_none=d['none']['Rhalf'],
                                            I2_none=d['none']['I2'])
    inv['table2_probe25_float32'] = rows
    s5 = J(os.path.join('scope', 's5_reverify.json'))
    inv['s5_reverify_float64'] = {r['model']: {m: dict(additive=v['additive'], maxNS=v['maxNS'], Rhalf=v['Rhalf'],
                                                       I2=v['I2'], corr_min=v['corrNS_min'], n=v['n'])
                                               for m, v in r['modes'].items()} for r in s5}
    w = 0.0
    for r in s5:
        for v in r['modes'].values():
            if v['additive']:
                w = max(w, v['maxNS'], v['Rhalf'], v['I2'])
    inv['s5_reverify_worst_additive_residual'] = w
    # (ii) 1,000 classes, T = 0, probe29
    p = J('probe29_fan200.json')
    for k in ('ship_lse0.0', 'lse0.0'):
        r = p[k]
        inv[f'probe29_{k}'] = dict(corrNS=r['corrNS'], corrNS_ci=r['corrNS_ci'], Rhalf=r['Rhalf'],
                                   I2=r['I2'], spNS=r['spNS'], del_N_minus_del_S=r['del_N'] - r['del_S'])
    # (iii) per-cell heads, probe1
    a = J('probe1_aggregator.json')
    inv['probe1_percell'] = {k: dict(corrNS=a[k]['corrNS'], one_minus_corr=1 - a[k]['corrNS'],
                                     Rhalf=a[k]['Rhalf'], I2=a[k]['I2'])
                             for k in ('gap_fc', 'ln_gap', 'tok_mlp')}
    inv['fig2d_1.4e-10_source'] = dict(
        file='the panel-(d) label of Figure 2',
        formula="max over (gap_fc, ln_gap, tok_mlp) of 1 - probe1_aggregator.json[k]['corrNS']",
        value=max(1 - a[k]['corrNS'] for k in ('gap_fc', 'ln_gap', 'tok_mlp')),
        argmax='ln_gap')
    # (iv) probe30 / probe31 / probe27b control values
    c30 = J('probe30_camfamily.json')
    inv['probe30_T0_corrNS'] = c30['corrNS']['0.0']
    inv['probe30_gradcam_max_dev'] = c30['cams']['grad-cam']['max_dev_over_all_T']
    return inv


def t8():
    rs = np.random.RandomState(0)
    a30 = rs.choice(10000, 128, replace=False); b30 = rs.choice(10000, 16, replace=False)
    p29 = np.random.RandomState(0).permutation(10000)[:224 + 48]
    i29, b29 = p29[:224], p29[224 + 16:224 + 48]
    c30 = J('probe30_camfamily.json'); p29j = J('probe29_fan200.json')
    T0 = {m: v['by_T']['0.0'] for m, v in c30['cams'].items()}
    same = sorted({m for m, v in T0.items() if abs(v['rho_with_necessity'] - T0['grad-cam']['rho_with_necessity']) < 1e-9})
    return dict(
        probe30_images_subset_of_probe29=bool(set(a30) <= set(i29)),
        probe30_images_equal_first128_of_probe29=bool(np.array_equal(a30, i29[:128])),
        bank_overlap=int(len(set(b30) & set(b29))), bank_sizes=[16, 32],
        probe30_T0_rho={m: v['rho_with_necessity'] for m, v in T0.items()},
        coinciding_maps=same,
        probe29_ship_T0_spGN=p29j['ship_lse0.0']['spGN'], probe29_ship_T0_spGS=p29j['ship_lse0.0']['spGS'],
        probe29_ship_T0_spGN_ci=p29j['ship_lse0.0']['spGN_ci'])


def main():
    res = dict(t5=t5(), t7=t7(), t8=t8())
    L.save('t5_t7_t8_misc.json', res)
    print(json.dumps(res['t5'], indent=1, default=float)[:3000])
    print(json.dumps(res['t8'], indent=1, default=float))


if __name__ == '__main__':
    main()
