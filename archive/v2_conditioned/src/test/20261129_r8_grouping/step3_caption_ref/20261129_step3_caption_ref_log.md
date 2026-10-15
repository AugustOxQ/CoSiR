# Log: step 3 caption reference (round 8)

- 2026-10-12 18:24 Wrote `r8_caption_ref.py`. First run of T and R0 (about 1 minute each). A reproduced round 7 exactly.
- 2026-10-12 18:28 Found the image posterior differs by up to 3.1e-6 within a painting (float noise). Replaced the exact-equality check with a 1e-5 tolerance plus an argmax check; reran T and R0 (numbers unchanged). Merged to `results.json`; wrote `step3_caption_ref.md`.
- No existing source file was edited. Round 7's `results/` was only read.
