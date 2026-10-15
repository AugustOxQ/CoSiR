"""Offline NLLB-200 English translation, whole-language shards and checked merge."""

import argparse
import hashlib
import json
import os
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL = "facebook/nllb-200-3.3B"
CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"
COLUMNS = "image_id genre emotion caption art_style painting language image_name split".split()
OUTPUT = "csv_row painting language nllb_src proxy source translation".split()
# CSV spelling is intentional. NLLB has no Southern Ndebele or Makhuwa token.
LANGUAGES = {
    "Burmese": ("mya_Mymr", 7360), "Darija": ("ary_Arab", 6341),
    "Emakhuwa": ("swh_Latn", 1940), "Hausa": ("hau_Latn", 7457),
    "Hindi": ("hin_Deva", 7253), "Igbo": ("ibo_Latn", 7768),
    "Indonesian": ("ind_Latn", 6038), "IsiNdebele": ("zul_Latn", 2165),
    "IsiXhosa": ("xho_Latn", 1967), "IsiZulu": ("zul_Latn", 7982),
    "Kazakh": ("kaz_Cyrl", 2543), "Kinyarwanda": ("kin_Latn", 5831),
    "Korean": ("kor_Hang", 7671), "Kyrgyz": ("kir_Cyrl", 6417),
    "Malay": ("zsm_Latn", 7449), "Setswana": ("tsn_Latn", 6272),
    "Swahili": ("swh_Latn", 7319), "Tagalog": ("tgl_Latn", 8338),
    "Tamil": ("tam_Taml", 6985), "Thai": ("tha_Thai", 5249),
    "Turkish": ("tur_Latn", 7573), "Urdu": ("urd_Arab", 3189),
    "Uzbek": ("uzn_Latn", 7700), "Vietnamese": ("vie_Latn", 7962),
    "Yoruba": ("yor_Latn", 7459), "arabic": ("arb_Arab", 6067),
    "chinese": ("zho_Hans", 6935),
}
PROXIES = {"IsiNdebele", "Emakhuwa"}
GENERATION = dict(num_beams=4, max_new_tokens=256, do_sample=False)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def letters(text):
    chars = [c for c in text if c.isalpha()]
    non_latin = sum("LATIN" not in unicodedata.name(c, "") for c in chars)
    return non_latin / len(chars) if chars else 0.0


def read_csv(path):
    data = pd.read_csv(path, keep_default_na=False)
    assert list(data.columns) == COLUMNS, data.columns.tolist()
    assert len(data) == 173745, len(data)
    counts = data.language.value_counts().to_dict()
    assert counts == {**{k: v[1] for k, v in LANGUAGES.items()}, "english": 6515}, counts
    assert data.caption.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    data.insert(0, "csv_row", np.arange(len(data)))
    return data[data.language != "english"].copy()


def mapping_and_checks(data, tokenizer, overrides):
    mapping = {k: dict(code=v[0], proxy=k in PROXIES) for k, v in LANGUAGES.items()}
    for override in overrides:
        name, code = override.split("=", 1)
        assert name in mapping, name
        mapping[name] = dict(code=code, proxy=True)
    tokens = set(tokenizer.all_special_tokens)
    assert "eng_Latn" in tokens
    checks = {}
    for name, entry in mapping.items():
        assert entry["code"] in tokens, (name, entry)
        sample = data.loc[data.language == name, "caption"].iloc[0]
        scripts = sorted({unicodedata.name(c, "UNKNOWN").split()[0] for c in sample if c.isalpha()})
        checks[name] = dict(sample=sample, unicode_letter_families=scripts,
                            non_latin_letter_share=letters(sample))
        print(f"{name}: rows={LANGUAGES[name][1]} code={entry['code']} proxy={entry['proxy']} "
              f"sample_scripts={','.join(scripts)} sample={sample[:100]!r}")
    checks["chinese"]["code_reason"] = "First sample contains simplified 面积 (traditional 面積): zho_Hans."
    checks["Darija"]["code_reason"] = "First sample contains Arabic letters: ary_Arab."
    print(checks["chinese"]["code_reason"])
    print(checks["Darija"]["code_reason"])
    return mapping, checks


def shard_languages(nshards):
    assert nshards > 0
    groups, loads = [[] for _ in range(nshards)], [0] * nshards
    for name in sorted(LANGUAGES, key=lambda x: (-LANGUAGES[x][1], x)):
        shard = min(range(nshards), key=lambda i: (loads[i], i))
        groups[shard].append(name)
        loads[shard] += LANGUAGES[name][1]
    for i, names in enumerate(groups):
        print(f"shard {i}: rows={loads[i]} languages={','.join(names)}")
    return groups


def validate_rows(frame, data):
    assert frame.csv_row.is_unique
    assert frame.csv_row.isin(data.csv_row).all()
    expected = data.set_index("csv_row").loc[frame.csv_row]
    for col, source_col in [("painting", "painting"), ("language", "language"), ("source", "caption")]:
        assert frame[col].tolist() == expected[source_col].tolist(), col
    assert frame.translation.map(lambda x: isinstance(x, str)).all()
    assert frame.proxy.isin([True, False]).all()


def save_parquet(path, frame):
    frame = frame.sort_values("csv_row").reset_index(drop=True)
    frame.index = pd.Index(frame.csv_row, name="index")
    frame.to_parquet(path)


def merge(args, data):
    frames, metas, coverage = [], [], {}
    for folder in args.merge:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["limit_per_language"] is None, "Cannot merge smoke outputs as full translations"
        metas.append(meta)
        frames.append(pd.read_parquet(folder / "translations.parquet")[OUTPUT])
        report = json.loads((folder / "coverage.json").read_text())
        assert not coverage.keys() & report.keys(), "Language appears in multiple shards"
        coverage.update(report)
    first = metas[0]
    for meta in metas:
        for key in ["model", "snapshot_hash", "csv_sha256", "mapping", "generation", "nshards"]:
            assert meta[key] == first[key], key
    assert first["csv_sha256"] == hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    assert sorted(m["shard"] for m in metas) == list(range(first["nshards"]))
    frame = pd.concat(frames, ignore_index=True)
    validate_rows(frame, data)
    assert sorted(frame.csv_row.tolist()) == data.csv_row.tolist()
    assert set(coverage) == set(LANGUAGES)
    for name, (_, count) in LANGUAGES.items():
        assert coverage[name]["rows"] == count
        rows = frame[frame.language == name]
        assert len(rows) == count
        assert (rows.nllb_src == first["mapping"][name]["code"]).all()
        assert (rows.proxy == first["mapping"][name]["proxy"]).all()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "translations.parquet").exists(), out
    save_parquet(out / "translations.parquet", frame)
    write_json(out / "coverage.json", coverage)
    write_json(out / "meta.json", dict(first, merged_from=args.merge, shard=None,
                                       shard_meta=metas, rows=len(frame)))
    print(f"merged rows={len(frame)} into {out / 'translations.parquet'}")


def translate(args, data, snapshot, tokenizer, mapping, checks, groups):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM

    assert args.device != "cuda" or torch.cuda.is_available()
    assert args.batch_size > 0
    assert args.limit_per_language is None or args.limit_per_language > 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "translations.parquet").exists(), out
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=dtype)
    model.to(args.device).eval()
    load_seconds = time.monotonic() - started
    generation = dict(GENERATION, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
    records, coverage, timings = [], {}, {}
    for name in groups[args.shard]:
        language_start = time.monotonic()
        rows = data[data.language == name].head(args.limit_per_language)
        tokenizer.src_lang = mapping[name]["code"]
        encoded = tokenizer(rows.caption.tolist(), add_special_tokens=True, truncation=False)["input_ids"]
        order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]), int(rows.iloc[i].csv_row)))
        assert max(map(len, encoded)) <= model.config.max_position_embeddings, name
        outputs, hits = {}, 0
        with torch.inference_mode():
            for start in range(0, len(order), args.batch_size):
                positions = order[start:start + args.batch_size]
                batch = tokenizer.pad({"input_ids": [encoded[i] for i in positions]},
                                      padding=True, return_tensors="pt").to(args.device)
                generated = model.generate(**batch, **generation)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for position, tokens, translation in zip(positions, generated.tolist(), decoded):
                    # Decoder start is excluded; EOS ends the actual generated sequence.
                    tokens = tokens[1:]
                    length = tokens.index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in tokens else len(tokens)
                    hits += int(length == GENERATION["max_new_tokens"])
                    outputs[position] = translation.strip()
        texts = [outputs[i] for i in range(len(rows))]
        for row, translation in zip(rows.itertuples(index=False), texts):
            records.append(dict(csv_row=row.csv_row, painting=row.painting, language=name,
                                nllb_src=mapping[name]["code"], proxy=mapping[name]["proxy"],
                                source=row.caption, translation=translation))
        shares = [letters(x) for x in texts]
        coverage[name] = dict(rows=len(rows), **mapping[name], empty_outputs=sum(not x for x in texts),
                              outputs_equal_to_source=sum(a == b.strip() for a, b in zip(texts, rows.caption)),
                              outputs_still_in_source_script=sum(x > 0.5 for x in shares),
                              output_non_latin_letter_share=float(np.mean(shares)),
                              hit_max_new_tokens=hits, median_source_length=float(np.median(rows.caption.str.len())),
                              median_output_length=float(np.median([len(x) for x in texts])))
        timings[name] = time.monotonic() - language_start
        print(f"translated {name}: rows={len(rows)} seconds={timings[name]:.1f}", flush=True)
    frame = pd.DataFrame(records, columns=OUTPUT)
    validate_rows(frame, data)
    selected = data[data.language.isin(groups[args.shard])]
    if args.limit_per_language is not None:
        selected = selected.groupby("language", sort=False).head(args.limit_per_language)
    assert sorted(frame.csv_row.tolist()) == sorted(selected.csv_row.tolist())
    save_parquet(out / "translations.parquet", frame)
    write_json(out / "coverage.json", coverage)
    write_json(out / "meta.json", dict(model=MODEL, snapshot_hash=Path(snapshot).name, csv=str(Path(args.csv).resolve()),
        csv_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(), mapping=mapping, script_checks=checks,
        generation=generation, batch_size=args.batch_size, device=args.device, dtype=str(dtype), seed=0,
        target_language="eng_Latn", source_truncation=False,
        model_generation_config=model.generation_config.to_dict(),
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, pandas=pd.__version__),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=True, ordering="token length, then csv_row, within greedy whole-language shards",
        shard=args.shard, nshards=args.nshards, shards=groups, limit_per_language=args.limit_per_language,
        rows=len(frame), length_unit="characters", source_script_threshold=0.5,
        source_script_measure="outputs_still_in_source_script counts outputs with >50% non-Latin letters; "
                              "non-Latin letter shares are reported for every language, including Latin sources",
        timings=dict(load_seconds=load_seconds, languages=timings, total_seconds=time.monotonic() - started)))
    print(f"saved rows={len(frame)} to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--out", default="/data/SSD2/pre_extract/artelingo28/nllb200_3.3b_en")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--proxy", action="append", default=[], metavar="NAME=CODE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-per-language", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--merge", nargs="+")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    data = read_csv(args.csv)
    print(f"non-English languages={data.language.nunique()} rows={len(data)} (English excluded=6515)")
    if args.merge:
        assert not args.dry_run
        merge(args, data)
        return
    assert 0 <= args.shard < args.nshards
    snapshot = snapshot_download(MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, tgt_lang="eng_Latn")
    mapping, checks = mapping_and_checks(data, tokenizer, args.proxy)
    groups = shard_languages(args.nshards)
    print(f"snapshot_hash={Path(snapshot).name}; selected shard={args.shard}")
    if not args.dry_run:
        translate(args, data, snapshot, tokenizer, mapping, checks, groups)


if __name__ == "__main__":
    main()
