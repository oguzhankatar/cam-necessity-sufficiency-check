"""The pre-normalisation (token) control at the 3x3 unit: the additive readout on the same tokens
(collapsed_pre_b3) against the separated 3x3 maps of the relocation arm (N_ln_b3, S_ln_b3), with
the tests and the Holm family of scripts/wsol/w8_control_test.py (which does the same at 1x1).
Writes out/l10_control_b3.json and prints the table."""
import io
import os
import sys
import contextlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'wsol'))
import wlib as W
import w8_control_test as W8

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
W8.CONTROLS = ['collapsed_pre_b3']
W8.SEPARATED = ['N_ln_b3', 'S_ln_b3']
saved = {}
W.save = lambda name, obj: saved.update(obj=obj)          # keep w8's own output file untouched
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    W8.main()
log = buf.getvalue()
print(log)
import json
with open(os.path.join(OUT, 'l10_control_b3.json'), 'w', encoding='utf-8') as f:
    json.dump(saved['obj'], f, indent=1)
with open(os.path.join(OUT, 'l10_control_b3.log'), 'w', encoding='utf-8') as f:
    f.write(log)
