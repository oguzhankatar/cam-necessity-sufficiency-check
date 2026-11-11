# The two runs of the class-token probe, and which block to read from which

Neither file supersedes the other. Read `part_a` from **h3_cls_A.json** and
`part_b` from **h3_cls.json**, which is what `scripts/verify_numbers.py` does.

## part_a -- the head site: read h3_cls_A.json

`h3_cls.json`'s `part_a` reports `vacuous: false` and `spatial_mass: 1.0` for
four of the eight models (`cait_xxs24_224`, `pit_s_224`, `convit_tiny`,
`xcit_tiny_12_p16_224`). That reading is an artefact of how prefix tokens were
counted: those four carry no separate prefix token, so the class token *is*
token 0 of the sequence and the probe attributed the whole score to spatial
mass.

`h3_cls_A.json` adds the diagnostic that settles it, `n_active` -- the number of
cells carrying a non-negligible share of the score. It is **1 for all eight
models**, `active_cells` is `[0]` in every case, and `vacuous` is therefore
`true` for all eight. The score rests on a single token whether or not that
token sits inside the spatial grid, which is why the article says the probe
at the head is exact and rests on one token of the 197.

## part_b -- the earlier-block sites: read h3_cls.json

The A run re-ran `part_a` only and its `part_b` is empty. The measurement the
article quotes -- 6 models x 24 sites, none additive, worst `maxNS` 1.269 and
smallest 0.6408 -- exists only in `h3_cls.json`.
