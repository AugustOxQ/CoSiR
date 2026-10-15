HIGH — Method of one-sided affect steering: design — Deck notes: “No evaluation label is used anywhere.”
Source: docs/reports/stage/2026-10-13_aff_model_card.md:341–342 (§5) says evaluation labels build and score episodes; :286–292 shows settings selected using development performance.
Fix: Say “No evaluation labels fit the groupings, heads, factors or reader; labelled development episodes select settings, and labels construct and score evaluation episodes.”

HIGH — Plan for more datasets — Deck notes: “every held read must start by 18 October,” including the later dataset plan.
Source: docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md:65–74 excludes stage-3 CUB/SemArt/Affection reads from r9; :304 time-boxes r9’s reads by 18 October, while :188 allows Affection data until 24 October.
Fix: Say “Complete r9’s backbone/language reads by 18 October; stage-3 dataset reads get their own decision and schedule; Affection data cutoff is 24 October.”

MEDIUM — Method of one-sided affect steering: training — Deck: “only the weights and the gate threshold are picked on development episodes, then frozen.”
Source: docs/reports/stage/2026-10-13_aff_model_card.md:151–156 (§3.4) also selects checkpoint A3 from six training runs using seed-42 R@1 and condition gain.
Fix: Include development selection of the factor checkpoint, and specify that the frozen settings apply to the held read.

MEDIUM — Test of one-sided affect steering on held paintings — Deck compares fresh +0.59 [+0.46, +0.73] and AFF–R1 +0.20 with held results without disclosing fresh-seed weight selection.
Source: docs/reports/stage/2026-10-13_aff_model_card.md:379–384 and docs/reports/stage/2026-10-12_affect_steering_line.md:290–314 identify fresh seeds 49–51 and re-picked weights; freezing seed-42 settings gives +0.59 [+0.46, +0.72] over B′(A0) and +0.09 [+0.01, +0.18] over R1.
Fix: Name fresh seeds 49–51 and their re-picked weights beside the comparison, or use the frozen-setting fresh results for comparison with held seeds 52–54.

MEDIUM — Test of one-sided affect steering on held paintings — Deck says “One read of paintings never used in tuning” but omits the source’s selection and prior-read disclosures.
Source: docs/reports/auto/v2/2026-11-25_artelingo_held_test.md:77–85 (§10.4 disclosures) requires reporting about 50 seed-42 variants plus about 10 later variants, earlier value-episode reads of held paintings, and near-duplicate training images.
Fix: Add a compact disclosure that AFF was selected after repeated development searches, these paintings had earlier value-task reads, and 24 held rows have a training-image cosine ≥0.99.

MEDIUM — Analysis of where the gain comes from — Deck: a random gate “scores the same, so the gain comes from steering the right side.”
Source: docs/reports/stage/2026-10-12_affect_steering_line.md:374–387 calls this a privileged mechanism control that knows the condition; the intervals show no measurable difference on fresh episodes, and the causal reading is explicitly an interpretation with no held control.
Fix: Say “A privileged random-side control was indistinguishable on fresh seeds 49–51; this suggests that steering one side accounts for most of the gain.”

MEDIUM — Analysis of where the gain comes from — Deck notes: the gate opens in “62% ... under the swapped condition, against 30% with 4 pairs,” without specifying the threshold.
Source: docs/reports/auto/v2/2026-12-03_kshot_supports.md:114–122 reports 61.9% versus 29.5% at τ₀; at τ₂ it is 57.1% versus 11.1%. AFF’s fused settings use both thresholds (model card:294–303).
Fix: Qualify these as seed-42 gate-open shares at the lower threshold τ₀, rather than the overall fused method’s shares.

MEDIUM — Analysis of style and genre — Deck presents round-7/8 scorer results, the told ceiling and AMIs without naming their development-only data.
Source: docs/reports/auto/v2/2026-11-26_design_l_style_source.md:26 and docs/reports/auto/v2/2026-11-29_learned_grouping.md:25–27,57 specify repeatedly used development seed 42, no held read, and no licensed method claim.
Fix: Label the slide “Development seed 42; exploratory, no held confirmation” and retain the distinction between the told ceiling and label-free candidates.

MEDIUM — Analysis of style and genre — Deck notes: DINOv2 “failed one of its checks fixed in advance.”
Source: docs/reports/auto/v2/2026-11-29_learned_grouping.md:81–85,120,150–153 requires both a positive pooled lower bound and emotion-pair non-inferiority; DINOv2’s pooled interval crosses zero and its emotion × style sum −63 misses the −49 bar.
Fix: Say “DINOv2 failed the positive pooled lower-bound check and the emotion × style non-inferiority check.”

MEDIUM — Discussion of learning without aspect labels — Deck: without labels, groups come from “models trained without labels.”
Source: docs/reports/stage/2026-10-13_aff_model_card.md:94–96,341–342 specifies an externally emotion-labelled GoEmotions classifier; .scratch/stage_pages/2026-10-14_aff_scope_limits.draft.md:113 says CSD used LAION style tags, as the deck’s own notes acknowledge.
Fix: Say “We fit groups without our evaluation aspect labels, while using pretrained encoders that can carry outside label supervision.”

MEDIUM — Experiment on two more backbones — Deck: “every part above the encoder is re-fitted.”
Source: docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md:162–171 explicitly carries the GoEmotions affect and affect-k-means groupings unchanged; image/caption groupings, heads, factors, reader and picks are re-fitted.
Fix: Specify which parts are re-fitted and that both GoEmotions-derived groupings remain fixed across backbones.

MEDIUM — Experiment on two more backbones — Deck: Qwen’s held read “is running” and “started this morning.”
Source: src/test/20261202_r9_scope/20261202_r9_scope_log.md:635–639 records H7’s launch/verdict, but ends with H6’s pre-read passing at 08:57; it contains no H6 held launch. The supplied source therefore does not confirm the claimed status.
Fix: Say “Qwen’s pre-read passed; held launch not confirmed in the supplied log,” or cite an updated launch record before saying it is running.

MEDIUM — Experiment on two more backbones — Deck: “groups under 100 rows now trigger a refit with the next seed.”
Source: src/test/20261202_r9_scope/20261202_r9_scope_log.md:558–559,627–628 records the a2 choice and SigLIP caption k-means restarting from seed 42 (smallest group 1) to seed 43 (smallest 939); neither this log nor the supplied r9 spec states the 100-row rule.
Fix: Cite the amended rule that establishes 100, or restrict the statement to the recorded singleton-triggered restart and its seed-43 outcome.

MEDIUM — Plan for more datasets — Deck: CUB development probes have “run, not yet read.”
Source: src/test/20261202_r9_scope/20261202_r9_scope_log.md:463–464 last reports SemArt done and CUB in progress, with no later CUB completion entry; :424–426 also records CUB-specific relaxation of the episode protocol.
Fix: Mark CUB completion as unconfirmed in the supplied log; when reporting completed probes, disclose that CUB omits the third-aspect control and lowers the value floor to 10 images.

LOW — Analysis of where the gain comes from — Deck identifies the k-shot table as “development episodes” and “the seed every setting was picked on,” without giving the seed number.
Source: docs/reports/auto/v2/2026-12-03_kshot_supports.md:12–13,67–71 and src/test/20261203_kshot_supports/results/ks_big_table.txt:1 specify 12,288 development episodes from seed 42.
Fix: Label the table “Development seed 42, 12,288 episodes; frozen four-pair settings.”

Numbers checked: 190 numeric occurrences in slide text, tables and notes (repetitions and interval endpoints counted; paths and model/score identifiers excluded); numeric annotations in all four figures also checked.
