"""TASK 4, run split for wall-clock: part A of t4_faithfulness.py for the
positive temperatures T = 4, 8 only (third process; t4b covers 0.5, 1, 2), in a second process, saved to
out/t4_faithfulness_posT2.json. Identical code (imported); the only difference
is that the random-ordering generator (torch, seed 0) starts at T = 4 instead
of continuing the stream from T = -8 ... 0, so the 16 random orderings per
image are different draws of the same distribution."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_faithfulness as T4

T4.TS = [4.0, 8.0]
_save = T4.L.save
T4.L.save = lambda name, obj: _save('t4_faithfulness_posT2.json', obj)

if __name__ == '__main__':
    T4.part_a()
