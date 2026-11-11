"""Tasks 3 and 4.

Task 3 -- bootstrap details of the localisation statistics, read from the code
(line numbers are located by searching wlib.py / w3_stats.py, never typed):
resample counts per metric, how the MaxBoxAccV2 bootstrap P is formed, and its
floor, plus how often the floor binds in the primary set.

Task 4 -- what changes the two verdicts in the image-level re-analysis
(analyses/localisation_reanalysis/out/t3_clustered.{json,log}): the test or the unit.
"""
import os
import re
import json
from collections import Counter
import llib as L

C3 = os.path.join(L.ROOT, 'analyses', 'localisation_reanalysis', 'out')


def find(path, pattern, first=True):
    hits = [i + 1 for i, ln in enumerate(open(path, encoding='utf-8')) if re.search(pattern, ln)]
    return hits[0] if first else hits


def block(path, start_pat, end_pat):
    a = find(path, start_pat)
    lines = open(path, encoding='utf-8').read().split('\n')
    for j in range(a, len(lines)):
        if re.search(end_pat, lines[j]):
            return a, j + 1
    return a, len(lines)


def main():
    wl = os.path.join(L.WS, 'wlib.py')
    w3 = os.path.join(L.WS, 'w3_stats.py')
    P = L.W.PROTOCOL
    loc = dict(
        wlib_n_boot=find(wl, r'^\s*n_boot=\d+'),
        wlib_n_boot_mbav2=find(wl, r'^\s*n_boot_mbav2=\d+'),
        wlib_seed=find(wl, r'^\s*seed=\d+'),
        wlib_boot_diff=block(wl, r'^def boot_diff', r'^\s*return d, '),
        wlib_boot_mbav2_diff=block(wl, r'^def boot_mbav2_diff', r'float\(min\(1\.0, max\(p, 1\.0 / n\)\)\)'),
        wlib_boot_mbav2_p_line=find(wl, r'p = 2\.0 \* min\(\(b <= 0\)\.mean\(\), \(b >= 0\)\.mean\(\)\)'),
        wlib_boot_mbav2_floor_line=find(wl, r'float\(min\(1\.0, max\(p, 1\.0 / n\)\)\)'),
        wlib_maxboxacc_v2=block(wl, r'^def maxboxacc_v2', r'return float\(np\.mean\(vals\)\), vals'),
        wlib_wilcoxon=block(wl, r'^def wilcoxon', r'rank_biserial=float'),
        wlib_mcnemar=block(wl, r'^def mcnemar', r'return dict\(n10=n10, n01=n01, p=p\)'),
        wlib_holm=block(wl, r'^def holm', r'return adj\.tolist\(\)'),
        w3_call_mbav2=find(w3, r'W\.boot_mbav2_diff\('),
        w3_call_boot_diff=find(w3, r'W\.boot_diff\(', first=False),
        w3_call_wilcoxon=find(w3, r'W\.wilcoxon\(', first=False),
        w3_call_mcnemar=find(w3, r'W\.mcnemar\('),
        w3_holm=find(w3, r'adj = W\.holm\('),
        w3_primary_const=find(w3, r'^PRIMARY = 8'),
    )
    # how often the 1/n floor binds among the primary MaxBoxAccV2 rows
    w3j = L.load_w3()
    floor = 1.0 / P['n_boot_mbav2']
    mb = [r for v in w3j.values() for m, rows in v['contrasts'].items() if m == 'MaxBoxAccV2'
          for r in rows if 'p_holm' in r]
    n_floor = sum(r['p'] <= floor + 1e-15 for r in mb)
    min_pholm = sorted(set(round(r['p_holm'], 12) for r in mb))
    per_metric = Counter(m for v in w3j.values() for m, rows in v['contrasts'].items()
                         for r in rows if 'p_holm' in r)
    tests = Counter((m, r['test']) for v in w3j.values() for m, rows in v['contrasts'].items()
                    for r in rows if 'p_holm' in r)
    t3 = dict(n_boot=P['n_boot'], n_boot_mbav2=P['n_boot_mbav2'], seed=P['seed'],
              ci='percentile 2.5 / 97.5', lines=loc, n_primary_mbav2=len(mb),
              n_primary_mbav2_at_floor=n_floor, floor=floor,
              distinct_p_holm_mbav2=min_pholm,
              primary_rows_per_metric=dict(per_metric),
              primary_tests={'%s|%s' % k: v for k, v in tests.items()})
    print('Task 3')
    print(json.dumps(t3, indent=1))

    # ---------------------------------------------------------------- task 4
    j = json.load(open(os.path.join(C3, 't3_clustered.json')))
    rows = j['rows']
    ch_img = [r for r in rows if r['sig_perm_img'] != r['sig_orig']]
    ch_rec = [r for r in rows if r['sig_perm_rec'] != r['sig_orig']]
    ch_clu = [r for r in rows if r['sig_perm_img'] != r['sig_perm_rec']]
    ch_cb = [r for r in rows if r['sig_cboot'] != r['sig_orig']]
    same = sorted((r['dataset'], r['backbone'], r['metric'], r['a'], r['b']) for r in ch_img) == \
        sorted((r['dataset'], r['backbone'], r['metric'], r['a'], r['b']) for r in ch_rec)
    keys = ('dataset', 'backbone', 'metric', 'a', 'b', 'diff', 'test_orig', 'p_orig', 'p_holm_orig',
            'sig_orig', 'p_perm_rec', 'p_holm_perm_rec', 'sig_perm_rec', 'p_perm_img',
            'p_holm_perm_img', 'sig_perm_img', 'p_cboot', 'p_holm_cboot', 'sig_cboot',
            'p_wil_img', 'p_holm_wil_img', 'sig_wil_img')
    detail = [{k: r.get(k) for k in keys} for r in ch_img]
    # the design note of C3 (number of flips etc.) is quoted from the json
    t4 = dict(n=len(rows), changed_image_perm=len(ch_img), changed_record_perm=len(ch_rec),
              changed_clustering_only=len(ch_clu), changed_cluster_boot=len(ch_cb),
              same_two_rows_image_vs_record=same, rows=detail,
              wil_img_changed=len([r for r in rows if r['sig_wil_img'] != r['sig_orig']]),
              summary=j['summary'], design=j['design'])
    print('\nTask 4')
    print('  n=%d; changed: image-level permutation %d, record-level permutation %d, '
          'record->image permutation (clustering alone) %d, image-clustered bootstrap %d, '
          'Wilcoxon on image means %d; same rows image vs record: %s'
          % (len(rows), len(ch_img), len(ch_rec), len(ch_clu), len(ch_cb),
             t4['wil_img_changed'], same))
    for d in detail:
        print('  %s %s %s %s vs %s diff %+.4f | %s p %.4g holm %.4g %s | rec-perm p %.4g holm %.4g %s'
              ' | img-perm p %.4g holm %.4g %s | cboot p %.4g holm %.4g %s'
              % (d['dataset'], d['backbone'], d['metric'], d['a'], d['b'], d['diff'],
                 d['test_orig'], d['p_orig'], d['p_holm_orig'], d['sig_orig'],
                 d['p_perm_rec'], d['p_holm_perm_rec'], d['sig_perm_rec'],
                 d['p_perm_img'], d['p_holm_perm_img'], d['sig_perm_img'],
                 d['p_cboot'], d['p_holm_cboot'], d['sig_cboot']))
    L.save('l3_bootstrap_clustered.json', dict(task3=t3, task4=t4))


if __name__ == '__main__':
    main()
