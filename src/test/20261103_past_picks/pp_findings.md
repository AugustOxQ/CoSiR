Gate writers: extraction store uses ok; parity, ii, iii, v, vi, viii use pass;
r14/r15/SemArt iv and r14/r15 vii use passed. Reduced replay never passes.
Node run status: node/cstar_<enc>.json and photo_<enc>.json, five seeds.
SemArt stage1 reads new records as committed: runner development wrapper reads local merged records only
while making validation entries. Final scoring still uses committed records.
SemArt partial records need a per-entry wrapper to keep healthy encoders independent.
