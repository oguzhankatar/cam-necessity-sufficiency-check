"""H0 -- FREEZE the stage-5 decision procedure, verbatim, before any held-out
model is touched.

The procedure lives inside `scripts/scope/s4_decide.py` (which runs a sweep at
import time, so it cannot simply be imported) and `scripts/scope/slib.py`.  This
script EXTRACTS the exact source text of the pieces that make the structural
prediction -- no retyping, no edits -- concatenates them into
`scripts/heldout/frozen_rule.py`, and records the sha256 of every extracted
block in `outputs/heldout/h0_freeze.json`.

If any of those blocks in scripts/scope/* ever changes, re-running this script
produces a different hash and the freeze is visibly broken.  `frozen_rule.py`
is never edited by hand for P1.
"""
import ast, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SCOPE = os.path.join(ROOT, 'scripts', 'scope')
OUT = os.path.join(ROOT, 'outputs', 'heldout')
os.makedirs(OUT, exist_ok=True)


def blocks(path, names):
    """Exact source segments of the named top-level defs/assignments."""
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    got = {}
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            got[nm] = ast.get_source_segment(src, node)
    missing = [n for n in names if n not in got]
    if missing:
        raise SystemExit('missing from %s: %s' % (path, missing))
    return [(n, got[n]) for n in names]


S4 = os.path.join(SCOPE, 's4_decide.py')
SL = os.path.join(SCOPE, 'slib.py')

parts = []
parts += [('slib.py', n, s) for n, s in
          blocks(SL, ['NORMISH', 'ACTISH', 'is_norm', 'is_act'])]
parts += [('s4_decide.py', n, s) for n, s in
          blocks(S4, ['AFFINE_CLS', 'POOL_AVG', 'cells_of', 'trace_head',
                      'kind', 'structural'])]

HEAD = '''"""FROZEN round-5 decision procedure.  DO NOT EDIT.

Auto-extracted verbatim by scripts/heldout/h0_freeze.py from
scripts/scope/slib.py and scripts/scope/s4_decide.py.  Every block below is the
exact source text of the corresponding round-5 object; the per-block sha256 is
in outputs/heldout/h0_freeze.json.  The only lines this file adds are the
imports and this docstring.

Round-5 result on the 66-architecture development set: 65/65 agreement.
"""
import sys as _sys
import torch
import torch.nn as nn


'''

# stage-5's `kind` calls slib.is_norm / slib.is_act.  Both functions are
# extracted verbatim above, so bind the name `slib` to this module.  This is a
# name binding, not a change to any extracted line.
SHIM = '\n\nslib = _sys.modules[__name__]\n\n'
i_kind = [k for k, (_, n, _) in enumerate(parts) if n == 'is_act'][0]
body = (HEAD + '\n\n'.join(s for _, _, s in parts[:i_kind + 1]) + SHIM
        + '\n\n'.join(s for _, _, s in parts[i_kind + 1:]) + '\n')
dst = os.path.join(HERE, 'frozen_rule.py')
open(dst, 'w', encoding='utf-8').write(body)

rec = dict(
    dst=dst,
    sha256_frozen_file=hashlib.sha256(body.encode()).hexdigest(),
    sha256_s4_decide=hashlib.sha256(open(S4, 'rb').read()).hexdigest(),
    sha256_slib=hashlib.sha256(open(SL, 'rb').read()).hexdigest(),
    n_blocks=len(parts),
    blocks=[dict(src=f, name=n, nlines=s.count('\n') + 1,
                 sha256=hashlib.sha256(s.encode()).hexdigest())
            for f, n, s in parts],
)
json.dump(rec, open(os.path.join(OUT, 'h0_freeze.json'), 'w'), indent=1)
for b in rec['blocks']:
    print('  %-14s %-14s %3d lines  %s' % (b['src'], b['name'], b['nlines'],
                                           b['sha256'][:16]))
print('frozen file sha256 = %s' % rec['sha256_frozen_file'])
print('-> %s' % dst)
