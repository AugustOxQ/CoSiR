# Findings and readings

No parallel ticket's ac_*.py source or output number was read. This check imports no
round ac_* module. Frozen r10 label/split helpers and r11 ranges are permitted inputs.
Raw store index, annotation header, frozen export shapes and frozen r11 numeric log
were consulted. No real held measurement was run.

Open readings (also in ac_vh.py's module comment):

- Affection inherits the one-aspect r11 read-out guard, gain >= -r.
- CRL version ties use <= .001 in the mean normalized score units; wording uses
  strictly more than .001*r. This resolves the three unequal ArtELingo r_a units.
- Zero selection ranges stop rather than inventing a denominator floor.
- Smoke: first N sorted keys (default 2100), same menu/k20, 200 draws for both
  bootstraps; accepts sliced full exports or smoke exports. Full rejects smoke.
- Float32 input normalization, float64 covariance/transform and ranking/QE/CRL,
  one float32 cast after W/CRL, covariance denominator n-1, no eigenvalue clipping,
  norm denominator floor 1e-12, inherited from frozen r11's numeric convention.
- Export hash serialization was unspecified. Accept only hashes of the exact
  expected ordered keys as default JSON, compact JSON, or newline-joined UTF-8
  with/without a trailing newline. Array hashes are SHA256 of contiguous bytes.
- Full label counts/ties always compare against the binding spec; stage0a gate
  items label_counts_{split}/label_ties_{split} compare too when present (the index
  leaves these names open). Stage0a's overall ok must be true.
- ArtELingo arm IDs are cstar/crl_clip_painting/crl_qwen_painting; fixed picks stay
  fixed while their cross-fit scores are remeasured.
- AMI SC readout fields are null. Extra descriptive fields are ignored; picks and
  SC ID sets are exact. Numerical comparisons are absolute <= 1e-6.

Integration with the stage runner is intentionally left for the later run chat;
no real input exists yet, and no producer implementation was inspected to resolve
these interface readings.
