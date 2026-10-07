# Run log: reader fix, round 5 (idea 3, GoEmotions placement of captions on AFF)

Times in Amsterdam local time (`TZ=Europe/Amsterdam date '+%F %H:%M'`). Folder date `20261123` is a sequence number.

| Time | Step |
|---|---|
| 2026-10-07 07:43 | Tab `idea3-goemotions` started from `docs/superpowers/handoffs/2026-10-07-idea3-goemotions-handoff.md`; reading list read; GPU free, load 0.3 |
| 07:45 to 08:04 | Handoff §5 open points settled with the user one at a time: not design L; GE head = logistic head on the 28 GoEmotions probabilities with the CLIP caption head's recipe; candidates G-T and G-TF; round 4's carry on seed 42, AUCs descriptive, test on 52 to 54 pre-registered; comparators B, B′(A0), B′_G, counterpart, B′(A1) beside; GoEmotions on the GPU under the lock if free, else CPU. Both design sections approved |
| 08:05 | Spec committed (393c3c2, SHA-256 bc738402…184ee) |
| 08:10 | The user approved the spec ("yeah go"); rule drafting dispatched to an Opus subagent (hashes and earlier rounds' constants only, no number of this round) |
| 09:33 | Draft rule written (747 lines); the controller read it in full and fixed the header time and the authorisation wording |
| 09:36 | Fresh Opus rule check dispatched (report `rule_check/opus_rule_check.md`) |
| 09:59 | Rule check done: 0 blocking, 5 should-fix (S1 sensitivity code owner and tests; S2 re-derivation phase-2 inputs; S3 G-TF's reader on test seeds only if carried; S4 positive check that the extension uses Q_GE; S5 precedence list restricted), 14 nits (among them the GoEmotions tolerance tightened to 1e-4 from a measured CPU-vs-CUDA maximum of 4.5e-6, and the disclosure of a percept-branch pilot of 2026-09-22 that had passed every `artelingo_train` caption through GoEmotions). It reproduced every constant bit for bit and computed no number of this round |
| 10:01 | All 19 findings applied by the controller; plan `docs/superpowers/plans/2026-10-07-idea3-goemotions.md` updated to match |
