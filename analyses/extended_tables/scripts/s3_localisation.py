"""S3 -- localisation additions to supplementary Tables S2/S3 and S6.

Sources (read only):
  analyses/localisation_reanalysis/out/t1_t2_contrasts.json
      rows_vs_collapsed, rows_vs_centre (the 192 primary contrasts, reproduced
      exactly from outputs/wsol/w3_stats.json) and task2.rows_vs_cam (the 96
      contrasts against plain CAM; Holm within (dataset, backbone, metric) over
      {separated maps} x {plain CAM, centre prior}).
  outputs/wsol/w2_{voc12seg,coco17}_{resnet50,vgg16gap,vitb16}.json  per_map.*.PxAP
Outputs: ../out/s3_localisation.json, ../out/s2_cam_rows.tex,
         ../out/s3_cam_rows.tex, ../out/s3_pxap_rows.tex
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *['..'] * 3))
C3 = os.path.join(ROOT, 'analyses', 'localisation_reanalysis', 'out', 't1_t2_contrasts.json')
WSOL = os.path.join(ROOT, 'outputs', 'wsol')
OUT = os.path.abspath(os.path.join(HERE, '..', 'out'))
os.makedirs(OUT, exist_ok=True)

MET = {'MaxBoxAccV2': 'MaxBoxAccV2', 'best_iou': 'best IoU', 'best IoU': 'best IoU',
       'pointing_strict': 'pointing, strict', 'pointing, strict': 'pointing, strict',
       'pointing_tol15': 'pointing, 15 px', 'pointing, 15 px': 'pointing, 15 px',
       'peak_mask_iou': 'peak-mask IoU', 'peak-mask IoU': 'peak-mask IoU'}
TEST = {'paired bootstrap': 'paired bootstrap', 'wilcoxon': 'Wilcoxon', 'Wilcoxon': 'Wilcoxon',
        'mcnemar': 'exact McNemar', 'exact McNemar': 'exact McNemar', 'exact_mcnemar': 'exact McNemar'}
ARM = lambda a: 'relocated normalisation' if a.endswith('_ln') else 'replaced aggregator'
HALF = lambda a: 'necessity' if a.startswith('N_') else 'sufficiency'
EXTENT = {'MaxBoxAccV2', 'best IoU', 'peak-mask IoU'}


def f4(x):
    return '+%.4f' % x if x >= 0 else '$-$%.4f' % -x


def f3(x):
    return '+%.3f' % x if x >= 0 else '$-$%.3f' % -x


def fp(p):
    if p == 0:
        return '$<$1e-300'
    if p < 1e-4:
        m, e = ('%.1e' % p).split('e')
        return '%se-%02d' % (m, abs(int(e)))
    return '%.3g' % p


def side(rows, sgn):
    d = np.array([r['diff'] for r in rows if r['sig'] == sgn])
    if len(d) == 0:
        return ''
    if len(d) == 1:
        return f3(d[0])
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    return '%s [%s, %s]' % (f3(q2), f3(q1), f3(q3))


def tally(rows):
    return dict(n=len(rows), win=sum(r['sig'] == 'win' for r in rows),
                loss=sum(r['sig'] == 'loss' for r in rows), tie=sum(r['sig'] == 'tie' for r in rows))


def norm(r):
    r = dict(r)
    r['metric'] = MET[r['metric']]
    r['test'] = TEST.get(r['test'], r['test'])
    r['arm'] = ARM(r['a'])
    return r


def main():
    D = json.load(open(C3))
    coll = [norm(r) for r in D['rows_vs_collapsed']]
    cent = [norm(r) for r in D['rows_vs_centre']]
    cam = [norm(r) for r in D['task2']['rows_vs_cam']]
    res = {}
    # ---- check S3's method on the collapsed rows (relocation: +0.047 [+0.021, +0.120], -0.015)
    rel = [r for r in coll if r['arm'] == 'relocated normalisation']
    res['check_S3_relocation_collapsed'] = [tally(rel), side(rel, 'win'), side(rel, 'loss')]
    # ---- tallies against plain CAM, by arm and backbone (S3 layout)
    s3 = []
    for arm in ('relocated normalisation', 'replaced aggregator'):
        A = [r for r in cam if r['arm'] == arm]
        groups = [('all', A)] + ([(bb, [r for r in A if r['backbone'] == bb])
                                  for bb in ('resnet50', 'vgg16gap', 'vitb16')]
                                 if arm == 'replaced aggregator' else [])
        for bb, R in groups:
            t = tally(R)
            s3.append(dict(arm=arm, backbone=bb, **t, med_win=side(R, 'win'), med_loss=side(R, 'loss')))
    res['cam_tally_total'] = tally(cam)
    res['cam_by_arm_backbone'] = s3
    # ---- survival of the wins over the collapsed map
    key = lambda r: (r['dataset'], r['backbone'], r['metric'], r['a'])
    camd = {key(r): r for r in cam}
    wins = [r for r in coll if r['sig'] == 'win']
    surv = [r for r in wins if camd[key(r)]['sig'] == 'win']
    res['wins_vs_collapsed'] = len(wins)
    res['of_which_win_vs_cam'] = len(surv)
    res['of_which_tie_vs_cam'] = sum(camd[key(r)]['sig'] == 'tie' for r in wins)
    res['of_which_loss_vs_cam'] = sum(camd[key(r)]['sig'] == 'loss' for r in wins)
    # ---- protocol-metric restriction (CUB: MaxBoxAccV2 only; VOC/COCO: pointing + peak-mask IoU)
    prot = [r for r in coll if r['dataset'] != 'cub' or r['metric'] == 'MaxBoxAccV2']
    extra = [r for r in coll if r['dataset'] == 'cub' and r['metric'] != 'MaxBoxAccV2']
    res['protocol_metrics_vs_collapsed'] = tally(prot)
    res['cub_extra_metrics_vs_collapsed'] = tally(extra)
    res['protocol_metrics_vs_centre'] = tally([r for r in cent if r['dataset'] != 'cub' or r['metric'] == 'MaxBoxAccV2'])
    res['protocol_metrics_vs_cam'] = tally([r for r in cam if r['dataset'] != 'cub' or r['metric'] == 'MaxBoxAccV2'])
    res['all_vs_collapsed'] = tally(coll)
    res['all_vs_centre'] = tally(cent)
    # ---- PxAP (secondary), from the run files
    px = []
    for ds in ('voc12seg', 'coco17'):
        for bb in ('resnet50', 'vgg16gap', 'vitb16'):
            pm = json.load(open(os.path.join(WSOL, 'w2_%s_%s.json' % (ds, bb))))['per_map']
            row = dict(dataset=ds, backbone=bb, **{k: pm[k]['PxAP'] for k in
                       ('collapsed', 'cam', 'center', 'N_lse', 'S_lse', 'N_ln', 'S_ln') if k in pm})
            for k in ('N_lse', 'S_lse', 'N_ln', 'S_ln'):
                if k in row:
                    row['d_' + k] = row[k] - row['collapsed']
                    row['dcam_' + k] = row[k] - row['cam']
            px.append(row)
    res['pxap'] = px
    above = [(r['dataset'], r['backbone'], k, round(r['d_' + k], 4)) for r in px
             for k in ('N_lse', 'S_lse', 'N_ln', 'S_ln') if ('d_' + k) in r and r['d_' + k] > 0]
    res['pxap_separated_above_collapsed'] = above
    res['pxap_ln_range'] = {k: [min(r['d_' + k] for r in px if 'd_' + k in r),
                                max(r['d_' + k] for r in px if 'd_' + k in r)] for k in ('N_ln', 'S_ln')}
    json.dump(res, open(os.path.join(OUT, 's3_localisation.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'pxap'}, indent=1, default=float))
    for r in px:
        print(r['dataset'], r['backbone'], {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})

    # ---- LaTeX: S2 block of the 96 plain-CAM contrasts, in S2's order and style
    DS = {'coco17': 0, 'cub': 1, 'voc12': 2, 'voc12seg': 3}
    cam_sorted = sorted(cam, key=lambda r: (DS[r['dataset']], r['backbone'], r['metric'], r['arm'], r['a'][0]))
    L = []
    for r in cam_sorted:
        L.append(' & '.join([r['dataset'], r['backbone'], r['metric'], r['arm'],
                             '%s vs plain CAM' % HALF(r['a']), '{:,}'.format(r['n']), r['test'],
                             '%.4f' % r['va'], '%.4f' % r['vb'],
                             '%s [%s, %s]' % (f4(r['diff']), f4(r['ci'][0]), f4(r['ci'][1])),
                             fp(r['p_holm']), r['sig']]) + r' \\')
    open(os.path.join(OUT, 's2_cam_rows.tex'), 'w').write('\n'.join(L) + '\n')
    # ---- LaTeX: S3 rows for plain CAM
    L = []
    for r in s3:
        lab = r['arm'] if r['backbone'] == 'all' else ''
        comp = 'plain CAM' if r['backbone'] == 'all' else ''
        L.append(' & '.join([lab, comp, r['backbone'], str(r['n']), str(r['win']), str(r['loss']),
                             str(r['tie']), r['med_win'], r['med_loss']]) + r' \\')
    open(os.path.join(OUT, 's3_cam_rows.tex'), 'w').write('\n'.join(L) + '\n')
    # ---- LaTeX: PxAP lines
    BB = {'resnet50': 'resnet50', 'vgg16gap': 'vgg16gap', 'vitb16': 'vitb16'}
    L = []
    for r in px:
        g = lambda k: (f3(r['d_' + k]) if ('d_' + k) in r else '')
        L.append(' & '.join([r['dataset'], BB[r['backbone']], '%.3f' % r['collapsed'], '%.3f' % r['cam'],
                             '%.3f' % r['center'], g('N_lse'), g('S_lse'), g('N_ln'), g('S_ln')]) + r' \\')
    open(os.path.join(OUT, 's3_pxap_rows.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
