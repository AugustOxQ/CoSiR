# Handoff: wait for r26's PercepT (CLIP L/14) files (about 21:15), score Table 3's last row; meanwhile follow the user

## Brief
- Done: intro skeleton (reference only, user's three choices); polish of Sections 2 and 4 with the user's a to d.
- State: Table 3 still has one empty row, PercepT on CLIP L/14. r26 sends its files at about 21:15.
- Next: tell the "CoSiR r26 run" tab this tab's name, then work on what the user asks. Score the PercepT row when r26's message arrives.
- Blocked: the user's 11:00 meeting. Ask nothing about it until the user says it is over.

> Written 2026-11-11 10:05 by the "paper sec4 tables 6" tab, cut at about 200k tokens at the user's yes. Outside the
> research loop: no loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`;
> REPO means `PF/CVPR2027-CoSiR-Paper/`; T3 means `/project/CoSiR/src/test/20261111_table3_rows/`.

## 1. Read in this order

1. This handoff, all of it.
2. `2026-11-11-paper-sec4-tables-6-handoff.md`, sections 2 to 5. They stand unchanged, except what section 3 here
   adds. Its section 4 items 2 and 3 (the PercepT row and the 4.1 note) are still the main job.
3. `PF/notes/intro_skeleton.md`, when the PercepT row is scored or the user works on the introduction.
4. `/project/CoSiR/.duck/reports/2026-11-11-polish-2.md`, only when the user returns to the polish or the captions.
5. `T3/20261111_table3_rows_log.md`, only when you score the PercepT row.
6. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of the earlier handoffs stands. New, all 2026-11-11, recorded in the PF README decisions table:

- **Intro skeleton** (`PF/notes/intro_skeleton.md`, reference only; the user writes the introduction): open on
  conditional similarity; the results paragraph claims "better than its teacher and the other compared methods" and
  names MMArt, smaller encoders and photos (user, after seeing the recommendation).
- **Polish of Sections 2 and 4** (Ducktor paper-polish run by hand, CBMI paper tone; REPO `03f320f`, `5cfde4b`):
  a. 4.1 seed sentence: "The results of CoSiR, its variants and PercepT are averages over five models ...".
  b. One protocol line in 4.1 ("Every table follows this protocol unless its caption states otherwise"); the captions
     of Tables 5 to 8 keep only what differs. Tables 2 and 3 keep full captions.
  c. CLAY in 4.1 uses Section 2's verb: "modulates frozen CLIP image embeddings with texts about the aspect".
  d. The bootstrap-interval sentence in 4.1 stays, though no table shows an interval yet (open with Table 5's bold and
     intervals).

## 3. Where things stand

- REPO (pushed) up to `5cfde4b`. PF (pushed) up to `3a52ddf` (trace `notes/trace/4_experiments.md` has a "Polish of
  2026-11-11" section). Ducktor records (gitignored, local): `.duck/reports/2026-11-11-polish-2.md`, one line each in
  `.duck/decisions.md` and `.duck/notebook.md`.
- **Table 3, PercepT (CLIP L/14):** unchanged from the tables-6 handoff. r26 confirmed (cross-session message at about
  09:47): files `OUT/test/k40/L1__s{seed}.npz` (keys `prob`, `names`, r17's test order), `OUT =
  /data/SSD2/cosir_r26/20261111_percept_original/outputs`, plus the k40 seeds that did not collapse with their count,
  at about 21:15. It sends to the highest "sec4 tables N" tab.
- **After scoring the row:** check the intro skeleton's paragraph 6 claim ("better than its teacher and the other
  compared methods") and the 4.2 sentence "For PercepT, the fair comparison is on the same backbone, Qwen3"; tell the
  user if the new row changes either.
- **Layout:** with the user's growing introduction the paper is 10 pages; Method starts on page 2, Figures 3 and 4 sit
  on page 8 after the references. A layout point for when the introduction is done.
- **The user's own open edits** (uncommitted, not ours): `Sections/1_introduction.tex` (introduction in progress),
  `Sections/2_related_work.tex` ("Qwen family" in the VLMs paragraph, `\TODO{do we add lens dataset?}` after the
  datasets paragraph). Commit them only when the user asks.

## 4. Open points for the next step

Key:
1. **First, message the "CoSiR r26 run" tab** with this tab's name (`ListAgents` prints it). Check
   `ReadNotifications` too.
2. **PercepT (CLIP L/14) row of Table 3** and **the 4.1 PercepT note**: as in the tables-6 handoff, section 4 items 2
   and 3.

Minor:
3. A possible 4.0 opening sentence was offered (Ducktor report): "We evaluate CoSiR by clustering and retrieval under
   each aspect, first on ArtELingo and then on MMArt, other encoders and photos." The user has not chosen it.
4. Figure captions and placement, the MMArt TODO under 4.4, Table 5 bold and intervals: when the user raises them.

## 5. Code and pitfalls

- All pitfalls of the earlier handoffs' section 5 stand (two chats share REPO and PF; commit only your own paths; no
  notes of ours in REPO; paper names only; no semicolons between clauses; "test set", never "held-out").
- **The user edits REPO files while we work.** When a file holds both their uncommitted lines and ours, stage only ours:
  write the HEAD-based version to the scratchpad, `git hash-object -w` it, `git update-index --cacheinfo`, then
  `git commit` with no paths (paths would commit the working tree, their lines included).
- In zsh, `echo ====` fails (`=` expansion): quote such strings.
- Ducktor is off in CoSiR chats. When the user asks for a Ducktor command, follow
  `~/.claude/plugins/marketplaces/ducktor/skills/duck-<name>/SKILL.md` by hand (it allows this), with the library
  skill it names; records go in `/project/CoSiR/.duck/`.

## 6. State at handoff

- Running: nothing of this tab. Others: r21 run, r26 run (GPU from 19:00).
- Uncommitted: in REPO, the user's own edits (section 3). In PF, `CBMI_IMP_CAMREADY (2).pdf` (the user's, left out of
  git). In the main repo, files not ours.
- On disk over 1 GB: nothing. The session scratchpad holds a few MB of page renders and logs.
