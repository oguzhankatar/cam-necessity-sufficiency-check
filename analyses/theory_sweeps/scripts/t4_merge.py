"""TASK 4 merge: part A from three processes (t4_faithfulness.json for T <= 0,
t4_faithfulness_posT.json for T = 0.5, 1, 2, t4_faithfulness_posT2.json for
T = 4, 8) plus part B (probe 29 normalised), into t4_faithfulness_merged.json,
and a compact summary table."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L
import t4_faithfulness as T4


def main():
    A = {}
    for f in ('t4_faithfulness.json', 't4_faithfulness_posT.json', 't4_faithfulness_posT2.json'):
        p = os.path.join(L.OUT, f)
        if os.path.exists(p):
            for T, r in json.load(open(p))['probe7_rerun'].items():
                r['source_file'] = f
                A.setdefault(T, r)
    A = {str(T): A[str(T)] for T in T4.TS if str(T) in A}
    B = T4.part_b()
    missing = [T for T in T4.TS if str(T) not in A]
    L.save('t4_faithfulness_merged.json', {'probe7_rerun': A, 'probe29_normalised': B, 'missing_T': missing})
    print('missing T:', missing)
    print('T     acc    | delnorm N  S    G    C0   CAM | insnorm N  S    G    C0   CAM | within del N-C0 [CI] | within ins N-C0 [CI]')
    for T, r in A.items():
        print(f"{float(T):+5.1f} {r['acc']:.4f} | " + ' '.join(f"{r['delnorm_' + t]:.3f}" for t in T4.TAGS) + ' | '
              + ' '.join(f"{r['insnorm_' + t]:.3f}" for t in T4.TAGS)
              + f" | {r['within_del_N_minus_C0']:+.3f} [{r['within_del_N_minus_C0_ci'][0]:+.3f},{r['within_del_N_minus_C0_ci'][1]:+.3f}]"
              + f" | {r['within_ins_N_minus_C0']:+.3f} [{r['within_ins_N_minus_C0_ci'][0]:+.3f},{r['within_ins_N_minus_C0_ci'][1]:+.3f}]"
              + f" | arch del_rand {r['archived']['del_rand']:.3f} new {r['del_rand']:.3f} ins_rand {r['ins_rand']:.3f}")
    for lab, rows in B.items():
        print('probe29', lab)
        for T, r in rows.items():
            print(f"  T={float(T):+4.0f} acc={r['acc']:.4f} rand del {r['del_rand']:.3f} ins {r['ins_rand']:.3f} | delnorm N {r['delnorm_N']:.3f} "
                  f"S {r['delnorm_S']:.3f} G {r['delnorm_G']:.3f} | insnorm N {r['insnorm_N']:.3f} S {r['insnorm_S']:.3f} G {r['insnorm_G']:.3f}")


if __name__ == '__main__':
    main()
