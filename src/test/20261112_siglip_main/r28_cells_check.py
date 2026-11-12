"""r28 job H: check paper_cells.md against its sources.

Every value cell in the sheet (a column whose header starts with "SigLIP 1") is paired with the next
column (its source). A source is one or more refs joined by " ; ":
  ALIAS/file.json :: path/with/{a}/{k}/{i}         (dict path; [..] list index with {i})
  ALIAS/file.json :: id=<cell id with {a}/{k}> => field   (verdict / label cell lists)
{a}, {i} expand over emotion, style, genre (a ref that uses them gives three values); {k} is the class count.
"now" cells ("L<n>: v / v / v") are checked against the numbers on line n of the paper .tex file.
Run:  PYTHONDONTWRITEBYTECODE=1 python r28_cells_check.py [--values-only]   (exit 0 and "PASS" when all match)
"""
import json, re, sys, os

SRC = "/project/CoSiR/src/test"
ALIAS = {
    "r28": f"{SRC}/20261112_siglip_main/records",
    "r17": f"{SRC}/20261103_new_scores/records",
    "r18": f"{SRC}/20261103_outside_methods/records",
    "r20": f"{SRC}/20261111_encoder_photo_tables/records",
}
TEXDIR = "/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Tables"
SHEET = f"{SRC}/20261112_siglip_main/paper_cells.md"
ASPECTS = ["emotion", "style", "genre"]
KS = {"artelingo": [9, 27, 10]}
SKIP = {"main_clustering": 3, "main_retrieval": 3, "mmart": 1, "ablation": 1, "encoders": 1, "photos": 2, "cost": 4}
TEXNAME = {"T2": "main_clustering", "T3": "main_retrieval", "T4": "mmart", "T5": "ablation", "T6": "encoders",
           "T7": "photos", "T8": "cost"}
_cache = {}
# --values-only: check every value cell against its source and skip the "now" cells: they are the .tex of 2026-11-12
# about 20:00, and the paper chat has since put the SigLIP 1 values into the tables (Table 5 at 23:15).
VALUES_ONLY = "--values-only" in sys.argv


def load(path):
    if path not in _cache:
        _cache[path] = json.load(open(path))
    return _cache[path]


def fmt(v):
    return f"{v:.3f}"


def signed(v):
    """A change as the paper writes it: "+" before a gain, "0.000" for a change that rounds to zero."""
    s = fmt(v)
    return "0.000" if s in ("0.000", "-0.000") else (s if s.startswith("-") else "+" + s)


def norm(s):
    """The sheet writes changes as the paper does (Table 5): "+" before a gain and "0.000" for a change that
    rounds to zero; drop the "+" and the sign of a zero so both sides compare as fmt() writes them."""
    s = s.strip()
    s = s[1:] if s.startswith("+") else s
    return "0.000" if s == "-0.000" else s


def _walk(d, path):
    """path of "/"-joined keys; a key may itself contain "/" (clay_crl.json cells); "[n]" indexes a list"""
    segs = path.split("/")
    while segs:
        if isinstance(d, dict):
            for j in range(len(segs), 0, -1):
                key = "/".join(segs[:j])
                m = re.fullmatch(r"(.*)\[(\d+)\]", key)
                if key in d:
                    d, segs = d[key], segs[j:]
                    break
                if m and m.group(1) in d:
                    d, segs = d[m.group(1)][int(m.group(2))], segs[j:]
                    break
            else:
                raise KeyError(path)
        else:
            raise KeyError(path)
    return d


def resolve(ref):
    """ref -> list of floats"""
    ref = ref.strip().strip("`")
    fileref, spec = [s.strip() for s in ref.split("::", 1)]
    alias, fname = fileref.split("/", 1)
    d = load(os.path.join(ALIAS[alias], fname))
    multi = "{a}" in spec or "{i}" in spec
    out = []
    for i in (range(3) if multi else [None]):
        s = spec
        if i is not None:
            s = s.replace("{a}", ASPECTS[i]).replace("{i}", str(i)).replace("{k}", str(KS["artelingo"][i]))
        if s.startswith("id="):
            ident, field = [t.strip() for t in s[3:].split("=>")]
            cells = [c for c in d["cells"] if c["id"] == ident]
            assert len(cells) == 1, (ref, ident, len(cells))
            out.append(float(cells[0][field]))
        else:
            out.append(float(_walk(d, s)))
    return out


def tex_numbers(table, line):
    """numbers (as strings) and '--' of the value cells on a .tex line"""
    raw = open(os.path.join(TEXDIR, table + ".tex")).read().split("\n")[line - 1]
    raw = raw.replace("\\\\", "")
    cells = raw.split("&")[SKIP[table]:]
    out = []
    for c in cells:
        c = re.sub(r"\\(textbf|underline)\{", "", c).replace("}", "").strip()
        c = c.replace("$-$", "-").replace("$+$", "+")
        if re.fullmatch(r"[-+]?\d\.\d{3}", c) or c == "--":
            out.append(c)
    return out


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main():
    lines = open(SHEET).read().split("\n")
    table = None
    header = None
    n_value = n_now = n_text = 0
    bad = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"<!-- table: (\w+) -->", ln)
        if m:
            table = m.group(1)
            i += 1
            continue
        if table and ln.startswith("|") and header is None:
            header = split_row(ln)
            i += 2  # skip separator
            continue
        if table and ln.startswith("|") and header is not None:
            row = split_row(ln)
            offs = {}
            assert len(row) == len(header), (ln[:80], len(row), len(header))
            for j, h in enumerate(header):
                cell = row[j]
                if h.startswith("SigLIP 1"):
                    src = row[j + 1]
                    if src.startswith("none") or not re.search(r"::", src):
                        assert not re.search(r"\d\.\d{3}", cell), ("number without source", table, row[0], cell)
                        n_text += 1
                        continue
                    vals = []
                    for ref in src.split(" ; "):
                        vals += [fmt(v) for v in resolve(ref)]
                    if table == "T5" and "(removal)" in row[0]:      # changes: the exact paper sign style
                        got = [t.strip() for t in cell.split(" / ")]
                        vals = [signed(float(v)) for v in vals]
                    else:
                        got = [norm(t) for t in cell.split(" / ")]
                        vals = [norm(v) for v in vals]
                    n_value += len(got)
                    if got != vals:
                        bad.append((table, row[0], h, got, vals))
                elif h.startswith("now") and VALUES_ONLY:
                    continue  # the paper chat has moved the tables to SigLIP 1 since the snapshot
                elif h.startswith("now"):
                    mm = re.fullmatch(r"L(\d+): (.*)", row[j])
                    if not mm:
                        assert row[j].startswith("n/a"), (table, row[0], row[j])
                        continue
                    if table == "T8":
                        continue  # ms cells (one decimal) not re-checked
                    nums = tex_numbers(TEXNAME[table], int(mm.group(1)))
                    got = [t.strip() for t in mm.group(2).split(" / ")]
                    o = offs.get(mm.group(1), 0)
                    offs[mm.group(1)] = o + len(got)
                    n_now += len(got)
                    if got != nums[o: o + len(got)] or o + len(got) > len(nums):
                        bad.append((table, row[0], "now", got, nums))
            i += 1
            continue
        if table and header is not None and not ln.startswith("|"):
            header = None
            table = None
        i += 1
    print(f"value cells checked against source: {n_value}; text cells: {n_text}; 'now' cells checked against .tex: {n_now}")
    for b in bad:
        print("MISMATCH", b)
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
