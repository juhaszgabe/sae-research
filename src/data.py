"""Prompt sources and generators, the generation run, dataset assembly, balance, splits, validation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from . import model as model_lib
from .config import Config
from .labeling import label_frame
from .model import ModelBundle
from .utils import (
    JsonlCache, derive_seed, get_logger, normalize_text, paths, read_jsonl, read_parquet, write_json,
    write_jsonl, write_parquet,
)

log = get_logger("data")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GEN_COLUMNS = ["prompt_id", "formatted_prompt", "prompt_n_tokens", "response", "finish_reason", "response_n_tokens"]
_LABEL_COLUMNS = ["prompt_id", "label", "label_strict", "ambiguous", "evidence"]

STYLES = ("plain", "polite_verbose", "terse", "embedded", "roleplay", "hungarian")


# --------------------------------------------------------------------------- prompt records (§10.1)

def _prompt_id(behavior: str, source: str, base_id: str, style: str, language: str) -> str:
    return hashlib.sha1(f"{behavior}|{source}|{base_id}|{style}|{language}".encode("utf-8")).hexdigest()[:16]


def _record(behavior: str, source: str, base_id: str, text: str, category: str, group_id: str | None = None,
            style: str = "plain", language: str = "en", json_schema: dict | None = None,
            meta: dict | None = None) -> dict:
    base_id = str(base_id)
    return {
        "prompt_id": _prompt_id(behavior, source, base_id, style, language),
        "behavior": behavior,
        "text": text,
        "base_id": base_id,
        "group_id": group_id or f"{source}:{base_id}",
        "source": source,
        "prompt_category": category,
        "style": style,
        "language": language,
        "json_schema": json_schema,
        "meta": meta or {},
    }


def _base_key(source: str, base_id: str) -> str:
    """Identifies a base prompt across sources (base_id alone is only unique within a source)."""
    return f"{source}:{base_id}"


# --------------------------------------------------------------------------- prompt builders (§10.2)

def load_hf_source(name: str, spec: dict, seed: int) -> list[dict]:
    """datasets.load_dataset(hf_id, config, split). Verify text_field / category_field exist
    (else raise listing columns). Apply filter (equality), min/max chars; base_id = original row
    index; if max_n: sample with Random(derive_seed(seed, "source", name)) and re-sort by index.
    Returns dicts {source, base_id, text, prompt_category}."""
    import datasets

    ds = datasets.load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"])
    columns = list(ds.column_names)
    needed = [spec["text_field"], *([spec["category_field"]] if spec.get("category_field") else []),
              *(spec.get("filter") or {})]
    missing = [c for c in needed if c not in columns]
    if missing:
        raise KeyError(f"source {name!r} ({spec['hf_id']}): fields {missing} not found; available columns: {columns}")

    cat_map = spec.get("category_map") or {}
    rows = []
    for i, row in enumerate(ds):
        if any(row[k] != v for k, v in (spec.get("filter") or {}).items()):
            continue
        text = str(row[spec["text_field"]]).strip()
        if len(text) < spec.get("min_chars", 1) or len(text) > spec.get("max_chars", 10**9):
            continue
        if spec.get("category_field"):
            raw_cat = row[spec["category_field"]]
            if cat_map and raw_cat not in cat_map:
                raise KeyError(f"source {name!r}: category value {raw_cat!r} not in category_map {cat_map}")
            category = cat_map.get(raw_cat, str(raw_cat))
        else:
            category = spec["category"]
        rows.append({"source": name, "base_id": str(i), "text": text, "prompt_category": category})

    max_n = spec.get("max_n")
    if max_n and len(rows) > max_n:
        keep = sorted(random.Random(derive_seed(seed, "source", name)).sample(range(len(rows)), max_n))
        rows = [rows[i] for i in keep]
    log.info("source %s: %d prompts", name, len(rows))
    return rows


def build_refusal_prompts(cfg: Config, sources: Sequence[str] | None = None) -> list[dict]:
    """Default sources = cfg refusal_main_sources (AdvBench + Alpaca + XSTest).
    XSTest is included on purpose: safe-but-scary prompts the model sometimes refuses →
    behavior/category disagreement needed for the novelty claim."""
    sources = list(sources) if sources is not None else list(cfg["refusal_main_sources"])
    specs = cfg["refusal_sources"]
    prompts = []
    for name in sources:
        if name not in specs:
            raise KeyError(f"unknown refusal source {name!r} (available: {sorted(specs)})")
        for row in load_hf_source(name, specs[name], cfg["seed"]):
            prompts.append(_record("refusal", name, row["base_id"], row["text"], row["prompt_category"]))
    return sorted(prompts, key=lambda r: r["prompt_id"])


def _load_spec(spec_path: Path) -> dict:
    spec_path = Path(spec_path)
    if not spec_path.exists() and not spec_path.is_absolute():
        spec_path = _REPO_ROOT / spec_path
    with open(spec_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_OPTIONAL = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)


def _render(template: str, task: str, schema_json: str, pressure: str) -> str:
    text = _OPTIONAL.sub((lambda m: m.group(1)) if pressure else "", template)
    # The schema goes in last so that its braces are never treated as placeholders.
    return text.replace("{task}", task).replace("{pressure}", pressure).replace("{schema}", schema_json)


def build_format_prompts(cfg: Config, spec_path: Path = Path("configs/format_prompts.yaml")) -> list[dict]:
    """Cross phrasings × pressures × schemas; for each combination draw
    cfg format_topics_per_combo topics (Random(derive_seed(seed, "fmt", ph, pr, sc))); build text;
    sample cfg format_candidates records (derive_seed(seed, "fmt", "sample")); sort by prompt_id.
    Template syntax: {task}, {schema}, {pressure}; a [[...]] segment is dropped when pressure is
    empty, brackets always removed. Schema is inserted as json.dumps(schema, ensure_ascii=False)."""
    spec = _load_spec(spec_path)
    seed, n_topics = cfg["seed"], cfg["format_topics_per_combo"]
    prompts = []
    for ph, template in spec["phrasings"].items():
        for pr, pressure in spec["pressures"].items():
            for sc, schema_spec in spec["schemas"].items():
                schema = schema_spec["schema"]
                topics = random.Random(derive_seed(seed, "fmt", ph, pr, sc)).sample(spec["topics"], n_topics)
                for topic in topics:
                    text = _render(template, schema_spec["task"].replace("{topic}", topic),
                                   json.dumps(schema, ensure_ascii=False), pressure)
                    prompts.append(_record(
                        "format_break", "fmtgen", f"{ph}:{pr}:{sc}:{topic}", text, pr, group_id=f"fmt:{ph}",
                        json_schema=schema, meta={"phrasing_id": ph, "schema_id": sc, "topic": topic},
                    ))
    n = cfg["format_candidates"]
    if len(prompts) > n:
        prompts = random.Random(derive_seed(seed, "fmt", "sample")).sample(prompts, n)
    return sorted(prompts, key=lambda r: r["prompt_id"])


def _pseudo_word(rng: random.Random, syllables: list[str]) -> str:
    return "".join(rng.choice(syllables) for _ in range(rng.choice((2, 3)))).capitalize()


def build_hedging_prompts(cfg: Config, spec_path: Path = Path("configs/format_prompts.yaml")) -> list[dict]:
    """[TDK] known = TriviaQA questions; fictional = templates with seeded pseudo-names;
    future = templates with years after the model's knowledge cutoff."""
    spec = _load_spec(spec_path)["hedging"]
    seed = cfg["seed"]
    prompts = []

    known_spec = {**spec["triviaqa"], "category": "known", "max_n": spec["n_known"]}
    for row in load_hf_source("triviaqa", known_spec, seed):
        prompts.append(_record("hedging", "triviaqa", row["base_id"], row["text"], "known"))

    templates = spec["fictional_templates"]
    per_template = math.ceil(spec["n_fictional"] / len(templates))
    seen: set[str] = set()
    fictional = []
    for tid, template in templates.items():
        made = attempt = 0
        while made < per_template and attempt < per_template * 20:
            rng = random.Random(derive_seed(seed, "hedge", tid, attempt))
            attempt += 1
            syl = spec["name_syllables"]
            text = (template
                    .replace("{name}", f"{_pseudo_word(rng, syl)} {_pseudo_word(rng, syl)}")
                    .replace("{title}", f"The {rng.choice(spec['title_adjectives'])} {rng.choice(spec['title_nouns'])}")
                    .replace("{place}", _pseudo_word(rng, syl)))
            if text in seen:
                continue
            seen.add(text)
            fictional.append(_record("hedging", "hedgegen", f"{tid}:{made}", text, "fictional",
                                     group_id=f"hedge:{tid}", meta={"template_id": tid}))
            made += 1
    prompts.extend(fictional[:spec["n_fictional"]])

    future = [
        _record("hedging", "hedgegen", f"{tid}:{year}", template.replace("{year}", str(year)), "future",
                group_id=f"hedge:{tid}", meta={"template_id": tid, "year": year})
        for tid, template in spec["future_templates"].items() for year in spec["future_years"]
    ]
    if len(future) < spec["n_future"]:
        log.warning("only %d unique future prompts (templates × years) for n_future=%d", len(future), spec["n_future"])
    elif len(future) > spec["n_future"]:
        future = random.Random(derive_seed(seed, "hedge", "future")).sample(future, spec["n_future"])
    prompts.extend(future)
    return sorted(prompts, key=lambda r: r["prompt_id"])


def apply_style(text: str, style: str, translation: str | None = None) -> str:
    """Style wrappers for the shift pools (§10.5)."""
    if style == "plain":
        return text
    if style == "polite_verbose":
        return ("Hello! I hope you're having a great day. I have a question I've been thinking about "
                f"for a while: {text} Thank you so much in advance!")
    if style == "terse":
        return "q: " + re.sub(r"\s+", " ", text.lower()).strip().rstrip(".,;:!?")
    if style == "embedded":
        return f"Here is a message from a colleague that I need help with:\n\n---\n{text}\n---\n\nPlease respond to it."
    if style == "roleplay":
        return ("Let's do a creative writing exercise. You are a character in a novel, and another character "
                f"asks you: \"{text}\" Write your character's reply.")
    if style == "hungarian":
        if translation is None:
            raise ValueError("style 'hungarian' needs a translation")
        return translation
    raise ValueError(f"unknown style {style!r} (available: {STYLES})")


def build_ood_prompts(cfg: Config, behavior: str, shift: str, test_base_ids: set[str] | None) -> list[dict]:
    """style/language shifts wrap ONLY base prompts in the main test split (test_base_ids);
    source shift loads the listed sources. [TDK]

    `test_base_ids` holds "{source}:{base_id}" keys as returned by `data.test_base_ids`
    (None → read from the main dataset)."""
    shifts = cfg["shifts"]
    if shift not in shifts:
        raise KeyError(f"unknown shift {shift!r} (available: {sorted(shifts)})")
    spec = shifts[shift]
    if behavior not in spec["behaviors"]:
        raise ValueError(f"shift {shift!r} is not defined for behavior {behavior!r}")

    if spec["kind"] == "source":
        if behavior != "refusal":
            raise ValueError("source shifts are only defined for refusal")
        return build_refusal_prompts(cfg, sources=spec["sources"])

    if test_base_ids is None:
        test_base_ids = _main_test_keys(cfg, behavior)
    base = [p for p in load_prompts(paths(cfg, behavior).prompts)
            if p["style"] == "plain" and _base_key(p["source"], p["base_id"]) in test_base_ids]

    translations: dict[tuple[str, str, str], str] = {}
    if "hungarian" in spec["styles"]:
        tr_path = Path(cfg.data_root) / "raw" / "translations_hu.jsonl"
        if tr_path.exists():
            translations = {(r["behavior"], r["source"], str(r["base_id"])): r["text_hu"] for r in read_jsonl(tr_path)}
        else:
            log.warning("no translations file at %s", tr_path)

    prompts, n_missing = [], 0
    for style in spec["styles"]:
        for p in base:
            translation = None
            if style == "hungarian":
                translation = translations.get((behavior, p["source"], p["base_id"]))
                if translation is None:
                    n_missing += 1
                    continue
            language = "hu" if style == "hungarian" else p["language"]
            prompts.append({
                **p,
                "prompt_id": _prompt_id(behavior, p["source"], p["base_id"], style, language),
                "text": apply_style(p["text"], style, translation),
                "style": style,
                "language": language,
            })
    if n_missing:
        log.warning("shift %s: %d of %d base prompts skipped (no Hungarian translation)", shift, n_missing, len(base))
    return sorted(prompts, key=lambda r: r["prompt_id"])


def save_prompts(prompts: list[dict], path: Path) -> None:
    write_jsonl(path, prompts)


def load_prompts(path: Path) -> list[dict]:
    return read_jsonl(path)


# --------------------------------------------------------------------------- generation run (§10.6)

def _gen_params_hash(bundle: ModelBundle, max_new_tokens: int, batch_size: int) -> str:
    params = {
        "revision": bundle.revision,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "dtype": str(next(bundle.model.parameters()).dtype),
    }
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def run_generation(bundle: ModelBundle, cfg: Config, behavior: str, pool: str = "main",
                   prompts: list[dict] | None = None) -> pd.DataFrame:
    """Loads prompts (or uses the given list), skips prompt_ids already in the generations
    JsonlCache (cache key = prompt_id + sha1 of {revision, max_new_tokens, batch_size, dtype}),
    generates the rest batch by batch with model.generate (appending after every batch),
    returns a DataFrame of all generations for the pool."""
    p = paths(cfg, behavior, pool)
    prompts = prompts if prompts is not None else load_prompts(p.prompts)
    max_new = cfg.get(f"generation.max_new_tokens.{behavior}")
    batch_size = cfg.get("generation.batch_size")
    params_hash = _gen_params_hash(bundle, max_new, batch_size)
    cache = JsonlCache(p.generations)

    def key(prompt: dict) -> str:
        return f"{prompt['prompt_id']}:{params_hash}"

    todo = sorted((pr for pr in prompts if key(pr) not in cache), key=lambda pr: -len(pr["text"]))
    log.info("generation %s/%s: %d cached, %d to generate", behavior, pool, len(prompts) - len(todo), len(todo))
    for start in tqdm(range(0, len(todo), batch_size), desc=f"generate {behavior}/{pool}", disable=not todo):
        batch = todo[start:start + batch_size]
        outs = model_lib.generate(bundle, [pr["text"] for pr in batch], max_new, batch_size, show_progress=False)
        for pr, out in zip(batch, outs):
            cache.put(key(pr), {"prompt_id": pr["prompt_id"], "params_hash": params_hash, **out})

    return pd.DataFrame([{"prompt_id": pr["prompt_id"], **cache.get(key(pr))} for pr in prompts])


def load_generations(cfg: Config, behavior: str, pool: str = "main") -> pd.DataFrame:
    """All cached generations of a pool, one row per prompt_id (the most recent entry wins)."""
    latest: dict[str, dict] = {}
    for rec in read_jsonl(paths(cfg, behavior, pool).generations):
        latest[rec["value"]["prompt_id"]] = rec["value"]
    return pd.DataFrame(list(latest.values()))


# --------------------------------------------------------------------------- dataset assembly (§10.7)

def _json_or_none(value: Any) -> str | None:
    # Key order is preserved so that a stored schema re-serializes to the exact text in the prompt.
    return None if value is None else json.dumps(value, ensure_ascii=False)


def assemble(prompts: list[dict], generations: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join on prompt_id (raise listing missing ids). json_schema/meta stored as JSON strings."""
    df = pd.DataFrame(prompts)
    if df["prompt_id"].duplicated().any():
        raise ValueError(f"duplicate prompt_ids: {df.loc[df['prompt_id'].duplicated(), 'prompt_id'].tolist()[:10]}")
    df["json_schema"] = df["json_schema"].map(_json_or_none)
    df["meta"] = df["meta"].map(_json_or_none)
    for name, other in (("generations", generations), ("labels", labels)):
        missing = sorted(set(df["prompt_id"]) - set(other["prompt_id"]))
        if missing:
            raise KeyError(f"{len(missing)} prompt_ids missing from {name}: {missing[:20]}")
    gen = generations[[c for c in _GEN_COLUMNS if c in generations.columns]].drop_duplicates("prompt_id", keep="last")
    lab = labels[[c for c in _LABEL_COLUMNS if c in labels.columns]].drop_duplicates("prompt_id", keep="last")
    df = df.merge(gen, on="prompt_id", how="left", validate="one_to_one")
    df = df.merge(lab, on="prompt_id", how="left", validate="one_to_one")
    df["label"] = df["label"].astype("int64")
    df["ambiguous"] = df["ambiguous"].astype(bool)
    return df.sort_values("prompt_id").reset_index(drop=True)


def _trigrams(text: str) -> frozenset:
    words = text.split()
    if len(words) < 3:
        return frozenset([tuple(words)])
    return frozenset(zip(words, words[1:], words[2:]))


def _without_schema(text: str, json_schema: str | None) -> str:
    """Prompt text without the embedded JSON Schema. The schema is longer than the surrounding
    template, so with it every format prompt of one schema would count as a near-duplicate and the
    phrasing groups (§10.4) would collapse into one group per schema."""
    if not json_schema:
        return text
    return text.replace(json.dumps(json.loads(json_schema), ensure_ascii=False), " ")


def dedup_and_group(df: pd.DataFrame, jaccard: float = 0.7) -> pd.DataFrame:
    """Drop exact duplicates of normalize_text(text) (keep smallest prompt_id); union-find merge of
    group_ids for pairs with word-3-gram Jaccard ≥ jaccard (component takes the smallest group_id).
    For format prompts the embedded JSON Schema is left out of the 3-grams (see _without_schema)."""
    df = df.sort_values("prompt_id").reset_index(drop=True)
    norm = df["text"].map(normalize_text)
    n_before = len(df)
    df = df[~norm.duplicated(keep="first")].reset_index(drop=True)
    if len(df) < n_before:
        log.info("dropped %d exact duplicate prompts", n_before - len(df))

    grams = [_trigrams(normalize_text(_without_schema(t, js))) for t, js in zip(df["text"], df["json_schema"])]
    groups = df["group_id"].tolist()
    parent = {g: g for g in set(groups)}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    n_merged = 0
    for i in range(len(df)):
        gi, ni = grams[i], len(grams[i])
        for j in range(i + 1, len(df)):
            gj = grams[j]
            # |A∩B| / |A∪B| ≥ t requires min(|A|,|B|) / max(|A|,|B|) ≥ t.
            if min(ni, len(gj)) < jaccard * max(ni, len(gj)):
                continue
            inter = len(gi & gj)
            if inter >= jaccard * (ni + len(gj) - inter):
                a, b = find(groups[i]), find(groups[j])
                if a != b:
                    parent[max(a, b)] = min(a, b)
                    n_merged += 1
    if n_merged:
        log.info("merged %d group pairs by near-duplicate text (Jaccard ≥ %.2f)", n_merged, jaccard)
    df["group_id"] = [find(g) for g in groups]
    return df


def prevalence(df: pd.DataFrame, by: Sequence[str] = ("source", "prompt_category", "style")) -> pd.DataFrame:
    """Positive rate + counts overall and per column. Logs a warning if overall rate < 0.05 or > 0.95
    (plan: below 5 % the probe is unlearnable)."""
    def row(column: str, value: str, sub: pd.DataFrame) -> dict:
        out = {"by": column, "value": value, "n": len(sub), "n_pos": int(sub["label"].sum()),
               "pos_rate": float(sub["label"].mean()) if len(sub) else float("nan")}
        if "label_strict" in sub.columns and sub["label_strict"].notna().any():
            out["pos_rate_strict"] = float(sub["label_strict"].astype(float).mean())
        return out

    rows = [row("overall", "all", df)]
    for column in by:
        if column in df.columns:
            rows.extend(row(column, str(value), sub) for value, sub in df.groupby(column, sort=True))
    rate = rows[0]["pos_rate"]
    if not 0.05 <= rate <= 0.95:
        log.warning("overall positive rate %.3f is outside [0.05, 0.95]: the probe may be unlearnable", rate)
    return pd.DataFrame(rows)


def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    s = sum(weights.values())
    exact = {k: total * w / s for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - out[k]), k))
    for k in order[:total - sum(out.values())]:
        out[k] += 1
    return out


def balance(df: pd.DataFrame, target_n: int, pos_frac: float, tol: float, seed: int
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (selected, rest). If both classes suffice: sample round(target_n·pos_frac) positives
    and the rest negatives. Otherwise take the whole minority class and as many majority rows as
    keep the positive fraction within pos_frac ± tol (never more than target_n in total).
    Within a class, allocate across `source` proportionally (largest remainder), then sample with
    derive_seed(seed, "balance", class, source). Raise if minority < 20."""
    avail = {1: int((df["label"] == 1).sum()), 0: int((df["label"] == 0).sum())}
    if min(avail.values()) < 20:
        raise ValueError(f"minority class has {min(avail.values())} rows (< 20); counts: {avail}")
    want = {1: int(round(target_n * pos_frac))}
    want[0] = target_n - want[1]
    frac = {1: pos_frac, 0: 1 - pos_frac}

    short = [c for c in (1, 0) if avail[c] < want[c]]
    if not short:
        take = dict(want)
    else:
        minority = min(short, key=lambda c: avail[c] / want[c])
        majority = 1 - minority
        # minority share may drop to (its target fraction − tol) at most
        max_total = int(math.floor(avail[minority] / max(frac[minority] - tol, 1e-9) + 1e-9))
        take = {minority: avail[minority]}
        take[majority] = max(0, min(avail[majority], max_total - avail[minority], target_n - avail[minority]))

    chosen: list[str] = []
    for cls in (1, 0):
        sub = df[df["label"] == cls]
        counts = sub["source"].value_counts().to_dict()
        for source, n in _largest_remainder(counts, take[cls]).items():
            ids = sorted(sub.loc[sub["source"] == source, "prompt_id"])
            chosen.extend(random.Random(derive_seed(seed, "balance", cls, source)).sample(ids, n))

    mask = df["prompt_id"].isin(set(chosen))
    selected = df[mask].sort_values("prompt_id").reset_index(drop=True)
    rest = df[~mask].sort_values("prompt_id").reset_index(drop=True)
    return selected, rest


def make_splits(df: pd.DataFrame, test_frac: float, n_folds: int, seed: int) -> pd.DataFrame:
    """Adds `split` (train/test) and `fold` (0..n_folds-1 for train, -1 for test).
    Test = fold 0 of StratifiedGroupKFold(round(1/test_frac), shuffle=True,
    random_state=derive_seed(seed, "split")) on (label, group_id); remaining rows get folds from
    StratifiedGroupKFold(n_folds, shuffle=True, random_state=derive_seed(seed, "folds"))."""
    df = df.sort_values("prompt_id").reset_index(drop=True)
    y, groups = df["label"].to_numpy(), df["group_id"].to_numpy()

    outer = StratifiedGroupKFold(int(round(1 / test_frac)), shuffle=True, random_state=derive_seed(seed, "split"))
    train_idx, test_idx = next(outer.split(np.zeros(len(df)), y, groups))

    fold = np.full(len(df), -1, dtype=np.int64)
    inner = StratifiedGroupKFold(n_folds, shuffle=True, random_state=derive_seed(seed, "folds"))
    for f, (_, val) in enumerate(inner.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx])):
        fold[train_idx[val]] = f

    split = np.full(len(df), "train", dtype=object)
    split[test_idx] = "test"
    df["split"], df["fold"] = split, fold
    return df


def check_splits(df: pd.DataFrame, min_per_class_fold: int = 5, min_per_class_test: int = 10) -> None:
    """Raise if a group_id is in both train and test or in two folds, or class counts are too small."""
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    crossing = sorted(set(train["group_id"]) & set(test["group_id"]))
    if crossing:
        raise ValueError(f"{len(crossing)} group_ids in both train and test: {crossing[:10]}")
    multi = train.groupby("group_id")["fold"].nunique()
    if (multi > 1).any():
        raise ValueError(f"group_ids in more than one fold: {multi[multi > 1].index.tolist()[:10]}")
    if (train["fold"] < 0).any() or (test["fold"] != -1).any():
        raise ValueError("fold must be ≥ 0 for train rows and -1 for test rows")
    for cls in (0, 1):
        n_test = int((test["label"] == cls).sum())
        if n_test < min_per_class_test:
            raise ValueError(f"test split has {n_test} rows of class {cls} (< {min_per_class_test})")
        per_fold = train[train["label"] == cls].groupby("fold").size().reindex(sorted(train["fold"].unique()), fill_value=0)
        if (per_fold < min_per_class_fold).any():
            raise ValueError(f"class {cls} has < {min_per_class_fold} rows in some fold: {per_fold.to_dict()}")


def _inputs_from_disk(cfg: Config, behavior: str, pool: str, prompts, generations, labels):
    p = paths(cfg, behavior, pool)
    prompts = prompts if prompts is not None else load_prompts(p.prompts)
    generations = generations if generations is not None else load_generations(cfg, behavior, pool)
    if labels is None:
        labels = read_parquet(p.labels) if p.labels.exists() else label_frame(
            behavior, pd.DataFrame(prompts)[["prompt_id", "json_schema"]].merge(generations, on="prompt_id"))
    return prompts, generations, labels


def build_dataset(cfg: Config, behavior: str, prompts: list[dict] | None = None,
                  generations: pd.DataFrame | None = None, labels: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Full pipeline: assemble → drop ambiguous (if configured) → dedup_and_group → prevalence
    (saved as prevalence.csv) → balance → make_splits → check_splits. Writes main.parquet and
    unused.parquet (rest rows whose group_id is not in train; used as steering eval prompts).
    Inputs default to what is on disk. Returns the main DataFrame."""
    p = paths(cfg, behavior)
    ds = cfg["dataset"]
    prompts, generations, labels = _inputs_from_disk(cfg, behavior, "main", prompts, generations, labels)

    df = assemble(prompts, generations, labels)
    if ds["exclude_ambiguous"]:
        log.info("%s: dropping %d ambiguous rows of %d", behavior, int(df["ambiguous"].sum()), len(df))
        df = df[~df["ambiguous"]]
    df = dedup_and_group(df, ds["near_dup_jaccard"])

    p.dataset_dir.mkdir(parents=True, exist_ok=True)
    prevalence(df).to_csv(p.dataset_dir / "prevalence.csv", index=False)

    selected, rest = balance(df, ds["target_n"], ds["pos_frac"], ds["pos_frac_tol"], cfg["seed"])
    main = make_splits(selected, ds["test_frac"], ds["n_folds"], cfg["seed"])
    check_splits(main)

    train_groups = set(main.loc[main["split"] == "train", "group_id"])
    unused = rest[~rest["group_id"].isin(train_groups)].copy()
    unused["split"], unused["fold"] = "unused", -1

    write_parquet(main, p.dataset)
    write_parquet(unused, p.unused)
    log.info("%s: main dataset %d rows (%.2f positive), %d unused rows", behavior, len(main),
             main["label"].mean(), len(unused))
    return main


def build_ood_dataset(cfg: Config, behavior: str, shift: str) -> pd.DataFrame:
    """[TDK] assemble → drop ambiguous → drop rows whose normalized text is in main.parquet;
    no balancing; split = f"ood:{shift}", fold = -1. Writes ood_{shift}.parquet."""
    p = paths(cfg, behavior)
    prompts, generations, labels = _inputs_from_disk(cfg, behavior, f"ood_{shift}", None, None, None)
    df = assemble(prompts, generations, labels)
    if cfg.get("dataset.exclude_ambiguous"):
        df = df[~df["ambiguous"]]
    main_texts = set(read_parquet(p.dataset)["text"].map(normalize_text))
    df = df[~df["text"].map(normalize_text).isin(main_texts)].reset_index(drop=True)
    df["split"], df["fold"] = f"ood:{shift}", -1
    prevalence(df)
    write_parquet(df, p.ood(shift))
    return df


def load_dataset_df(cfg: Config, behavior: str, which: str = "main") -> pd.DataFrame:
    """which = "main" | "unused" | "ood_{shift}"."""
    return read_parquet(paths(cfg, behavior).dataset_dir / f"{which}.parquet")


def test_base_ids(cfg: Config, behavior: str) -> set[str]:
    """"{source}:{base_id}" keys of the main test split. base_id is only unique within a source,
    so the source is part of the key (a bare base_id would let train prompts into the OOD pools)."""
    return _main_test_keys(cfg, behavior)


def _main_test_keys(cfg: Config, behavior: str) -> set[str]:
    df = load_dataset_df(cfg, behavior)
    test = df[df["split"] == "test"]
    return {_base_key(s, b) for s, b in zip(test["source"], test["base_id"])}


# --------------------------------------------------------------------------- manual validation (§10.8)

def export_validation_sample(df_labeled: pd.DataFrame, out_dir: Path, n: int = 50, seed: int = 0) -> None:
    """20 label=1, 20 label=0, 10 ambiguous (shortfall split evenly). Writes annotate.csv
    (sample_id, prompt_text, response, human_label — NO automatic label, rows shuffled) and key.csv
    (sample_id, prompt_id, auto_label, ambiguous, evidence). human_label ∈ {1, 0, x}."""
    df = df_labeled.sort_values("prompt_id")
    amb = df["ambiguous"].astype(bool)
    pools = {
        "pos": df[(df["label"] == 1) & ~amb]["prompt_id"].tolist(),
        "neg": df[(df["label"] == 0) & ~amb]["prompt_id"].tolist(),
        "amb": df[amb]["prompt_id"].tolist(),
    }
    n_main = int(round(0.4 * n))
    quota = {"pos": n_main, "neg": n_main, "amb": n - 2 * n_main}
    while True:
        shortfall = sum(max(0, quota[k] - len(pools[k])) for k in quota)
        quota = {k: min(quota[k], len(pools[k])) for k in quota}
        spare = [k for k in quota if len(pools[k]) > quota[k]]
        if not shortfall or not spare:
            break
        for i in range(shortfall):
            quota[spare[i % len(spare)]] += 1

    rng = random.Random(derive_seed(seed, "validation"))
    chosen = [pid for k in ("pos", "neg", "amb") for pid in rng.sample(pools[k], quota[k])]
    rng.shuffle(chosen)
    sample = df.set_index("prompt_id").loc[chosen].reset_index()
    sample.insert(0, "sample_id", [f"s{i:03d}" for i in range(len(sample))])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotate = sample[["sample_id", "text", "response"]].rename(columns={"text": "prompt_text"})
    annotate["human_label"] = ""
    annotate.to_csv(out_dir / "annotate.csv", index=False, encoding="utf-8")
    key = sample[["sample_id", "prompt_id", "label", "ambiguous", "evidence"]].rename(columns={"label": "auto_label"})
    key.to_csv(out_dir / "key.csv", index=False, encoding="utf-8")


def score_validation(out_dir: Path) -> dict:
    """On rows with human_label ∈ {0,1}: n, agreement, Cohen's kappa, confusion matrix,
    precision/recall of the auto labeler, n_unclear, disagreements (list). Saves report.json;
    warns if kappa < 0.7."""
    from sklearn.metrics import cohen_kappa_score

    out_dir = Path(out_dir)
    ann = pd.read_csv(out_dir / "annotate.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(out_dir / "key.csv", dtype={"sample_id": str, "prompt_id": str})
    df = ann.merge(key, on="sample_id", validate="one_to_one")
    human_raw = df["human_label"].str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    invalid = sorted(set(human_raw) - {"0", "1", "x", ""})
    if invalid:
        raise ValueError(f"human_label values must be 1, 0 or x; found {invalid}")
    scored = df[human_raw.isin(["0", "1"])].copy()
    if scored.empty:
        raise ValueError("no rows with human_label in {0, 1}; fill in annotate.csv first")
    human = human_raw[scored.index].astype(int).to_numpy()
    auto = scored["auto_label"].astype(int).to_numpy()

    tp = int(((auto == 1) & (human == 1)).sum())
    tn = int(((auto == 0) & (human == 0)).sum())
    fp = int(((auto == 1) & (human == 0)).sum())
    fn = int(((auto == 0) & (human == 1)).sum())
    single_class = len(set(human) | set(auto)) < 2
    kappa = float("nan") if single_class else float(cohen_kappa_score(human, auto))
    report = {
        "n": int(len(scored)),
        "agreement": float((auto == human).mean()),
        "kappa": kappa,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "n_unclear": int((human_raw == "x").sum()),
        "n_unannotated": int((human_raw == "").sum()),
        "disagreements": [
            {"sample_id": s, "prompt_id": p, "auto_label": int(a), "human_label": int(h)}
            for s, p, a, h in zip(scored["sample_id"], scored["prompt_id"], auto, human) if a != h
        ],
    }
    if not single_class and kappa < 0.7:
        log.warning("validation kappa %.2f < 0.7: revise the labeler before trusting the labels", kappa)
    write_json(out_dir / "report.json", report)
    return report
