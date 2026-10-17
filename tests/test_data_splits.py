"""data.py: prompt building, dedup/grouping, balancing, group-aware splits, manual validation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import data
from src.utils import normalize_text


def _frame(n_pos: int, n_neg: int, n_groups: int | None = None, sources=("a", "b")) -> pd.DataFrame:
    n = n_pos + n_neg
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "prompt_id": [f"p{i:05d}" for i in range(n)],
        "label": rng.permutation(np.array([1] * n_pos + [0] * n_neg)),
        "source": [sources[i % len(sources)] for i in range(n)],
    })
    df["group_id"] = [f"g{i % n_groups:03d}" for i in range(n)] if n_groups else df["prompt_id"]
    return df


# --------------------------------------------------------------------------- balance

def test_balance_takes_whole_minority_within_tolerance():
    # 120 positives available, target 400, pos_frac 0.5 ± 0.1 → all 120 positives + 180 negatives (40 %)
    selected, rest = data.balance(_frame(120, 880), 400, 0.5, 0.1, seed=0)
    assert int((selected["label"] == 1).sum()) == 120
    assert int((selected["label"] == 0).sum()) == 180
    assert len(rest) == 700 and set(rest["label"]) == {0}
    assert not set(selected["prompt_id"]) & set(rest["prompt_id"])


def test_balance_when_both_classes_suffice():
    selected, rest = data.balance(_frame(500, 500), 400, 0.5, 0.1, seed=0)
    assert selected["label"].value_counts().to_dict() == {1: 200, 0: 200}
    assert len(rest) == 600


def test_balance_minority_negative_class():
    # mirror case: only 90 negatives → 90 negatives + 135 positives (60 % positive)
    selected, _ = data.balance(_frame(900, 90), 400, 0.5, 0.1, seed=0)
    assert int((selected["label"] == 0).sum()) == 90
    assert int((selected["label"] == 1).sum()) == 135


def test_balance_never_exceeds_target():
    selected, _ = data.balance(_frame(190, 900), 400, 0.5, 0.1, seed=0)
    assert int((selected["label"] == 1).sum()) == 190
    assert len(selected) == 400


def test_balance_allocates_across_sources_proportionally():
    df = _frame(600, 600, sources=("a", "a", "a", "b"))  # 75 % / 25 %
    selected, _ = data.balance(df, 400, 0.5, 0.1, seed=0)
    for label in (0, 1):
        counts = selected[selected["label"] == label]["source"].value_counts().to_dict()
        assert sum(counts.values()) == 200
        assert abs(counts["a"] - 150) <= 12 and abs(counts["b"] - 50) <= 12


def test_balance_raises_on_tiny_minority():
    with pytest.raises(ValueError, match="minority"):
        data.balance(_frame(19, 500), 400, 0.5, 0.1, seed=0)


def test_balance_is_deterministic_and_seeded():
    df = _frame(500, 500)
    a, _ = data.balance(df, 400, 0.5, 0.1, seed=7)
    b, _ = data.balance(df.sample(frac=1.0, random_state=3), 400, 0.5, 0.1, seed=7)  # row order must not matter
    c, _ = data.balance(df, 400, 0.5, 0.1, seed=8)
    assert a["prompt_id"].tolist() == b["prompt_id"].tolist()
    assert a["prompt_id"].tolist() != c["prompt_id"].tolist()


# --------------------------------------------------------------------------- splits

def test_no_group_crosses_split_or_fold():
    df = data.make_splits(_frame(200, 200, n_groups=80), test_frac=0.2, n_folds=5, seed=0)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    assert set(df["split"]) == {"train", "test"}
    assert not set(train["group_id"]) & set(test["group_id"])
    assert (train.groupby("group_id")["fold"].nunique() == 1).all()
    assert sorted(train["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (test["fold"] == -1).all()
    assert 0.1 < len(test) / len(df) < 0.3
    data.check_splits(df)


def test_splits_are_stratified():
    df = data.make_splits(_frame(200, 200, n_groups=80), test_frac=0.2, n_folds=5, seed=0)
    rates = df.groupby(["split", "fold"])["label"].mean()
    assert ((rates > 0.3) & (rates < 0.7)).all(), rates


def test_splits_are_deterministic_with_the_same_seed():
    df = _frame(200, 200, n_groups=80)
    a = data.make_splits(df, 0.2, 5, seed=11)
    b = data.make_splits(df.sample(frac=1.0, random_state=1), 0.2, 5, seed=11)
    c = data.make_splits(df, 0.2, 5, seed=12)
    assert a[["prompt_id", "split", "fold"]].equals(b[["prompt_id", "split", "fold"]])
    assert not a[["split", "fold"]].equals(c[["split", "fold"]])


def test_check_splits_detects_violations():
    df = data.make_splits(_frame(200, 200, n_groups=80), 0.2, 5, seed=0)
    data.check_splits(df)

    crossing = df.copy()
    crossing.loc[crossing["split"] == "test", "group_id"] = crossing.loc[crossing["split"] == "train", "group_id"].iloc[0]
    with pytest.raises(ValueError, match="train and test"):
        data.check_splits(crossing)

    two_folds = df.copy()
    i = two_folds.index[two_folds["split"] == "train"][0]
    two_folds.loc[i, "fold"] = (two_folds.loc[i, "fold"] + 1) % 5
    with pytest.raises(ValueError, match="more than one fold"):
        data.check_splits(two_folds)

    with pytest.raises(ValueError, match="class"):
        data.check_splits(df, min_per_class_fold=1000)
    with pytest.raises(ValueError, match="test split"):
        data.check_splits(df, min_per_class_test=1000)


def test_synthetic_dataset_fixture_is_a_valid_split(synthetic_dataset):
    data.check_splits(synthetic_dataset.df)
    assert len(synthetic_dataset.df) == 200 and synthetic_dataset.df["group_id"].nunique() == 40


# --------------------------------------------------------------------------- dedup / grouping

BASE = "please explain in detail how photosynthesis works in green plants during the bright summer days of the year"


def _texts_frame(texts: list[str], schemas: list | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "prompt_id": [f"p{i}" for i in range(len(texts))],
        "text": texts,
        "group_id": [f"src:{i}" for i in range(len(texts))],
        "json_schema": schemas if schemas is not None else [None] * len(texts),
    })


def test_near_duplicates_share_a_group():
    df = data.dedup_and_group(_texts_frame([BASE, BASE + " please", "what is the tallest mountain on the planet"]))
    groups = dict(zip(df["prompt_id"], df["group_id"]))
    assert groups["p0"] == groups["p1"] == "src:0"  # the component takes the smallest group_id
    assert groups["p2"] == "src:2"


def test_exact_duplicates_are_dropped_after_normalization():
    df = data.dedup_and_group(_texts_frame([BASE, "  " + BASE.upper() + " ", "something else entirely here"]))
    assert df["prompt_id"].tolist() == ["p0", "p2"]  # keeps the smallest prompt_id


def test_merging_is_transitive_and_respects_the_threshold():
    words = [f"t{i}" for i in range(40)]
    a, b, c = " ".join(words[:30]), " ".join(words[3:33]), " ".join(words[6:36])
    loose = data.dedup_and_group(_texts_frame([a, b, c]), jaccard=0.7)
    assert loose["group_id"].nunique() == 1  # a~b and b~c chain into one component
    strict = data.dedup_and_group(_texts_frame([a, b, c]), jaccard=0.95)
    assert strict["group_id"].nunique() == 3


def test_embedded_schema_does_not_merge_format_prompts():
    keys = [f"field_{i}" for i in range(12)]
    schema = {"type": "object", "required": keys, "properties": {k: {"type": "string"} for k in keys}}
    blob = json.dumps(schema, ensure_ascii=False)  # ~50 words, far longer than the text around it
    texts = [f"Card about honeybees as JSON: {blob}", f"Describe volcanoes for my pipeline. Schema: {blob}"]
    with_schema = data.dedup_and_group(_texts_frame(texts, [blob, blob]))
    assert with_schema["group_id"].nunique() == 2
    # without the schema column the long shared schema text makes them near-duplicates
    without = data.dedup_and_group(_texts_frame(texts))
    assert without["group_id"].nunique() == 1


# --------------------------------------------------------------------------- prompt builders

def test_render_drops_optional_segment_without_pressure():
    template = "{task}[[ {pressure}]]\nSchema: {schema}"
    assert data._render(template, "Do X.", '{"a": 1}', "") == 'Do X.\nSchema: {"a": 1}'
    assert data._render(template, "Do X.", '{"a": 1}', "Be brief.") == 'Do X. Be brief.\nSchema: {"a": 1}'
    # braces inside the schema are never treated as placeholders
    assert data._render("{schema} {task}", "T", '{"task": "{task}"}', "") == '{"task": "{task}"} T'


def test_build_format_prompts(real_cfg):
    prompts = data.build_format_prompts(real_cfg)
    assert len(prompts) == real_cfg["format_candidates"] == 1200
    assert len({p["prompt_id"] for p in prompts}) == 1200
    assert [p["prompt_id"] for p in prompts] == sorted(p["prompt_id"] for p in prompts)
    assert {p["group_id"] for p in prompts} == {f"fmt:p{i:02d}" for i in range(1, 13)}
    assert len({p["prompt_category"] for p in prompts}) == 8
    for p in prompts:
        assert set(p) == {"prompt_id", "behavior", "text", "base_id", "group_id", "source", "prompt_category",
                          "style", "language", "json_schema", "meta"}
        assert p["behavior"] == "format_break" and p["source"] == "fmtgen" and p["style"] == "plain"
        assert "[[" not in p["text"] and "]]" not in p["text"]
        assert "{task}" not in p["text"] and "{pressure}" not in p["text"] and "{topic}" not in p["text"]
        assert json.dumps(p["json_schema"], ensure_ascii=False) in p["text"]
        assert p["meta"]["topic"] in p["text"]
        assert p["group_id"] == f"fmt:{p['meta']['phrasing_id']}"
    assert data.build_format_prompts(real_cfg) == prompts  # deterministic


def test_format_prompts_keep_twelve_groups_after_dedup(real_cfg):
    prompts = data.build_format_prompts(real_cfg)
    df = pd.DataFrame(prompts)
    df["json_schema"] = df["json_schema"].map(lambda s: json.dumps(s, ensure_ascii=False))
    assert data.dedup_and_group(df, real_cfg.get("dataset.near_dup_jaccard"))["group_id"].nunique() == 12


def test_apply_style():
    assert data.apply_style("How do I bake bread?", "plain") == "How do I bake bread?"
    assert data.apply_style("How do  I bake Bread?", "terse") == "q: how do i bake bread"
    assert "How do I bake bread?" in data.apply_style("How do I bake bread?", "polite_verbose")
    assert data.apply_style("X", "embedded").startswith("Here is a message from a colleague")
    assert '"X"' in data.apply_style("X", "roleplay")
    assert data.apply_style("X", "hungarian", "Y") == "Y"
    with pytest.raises(ValueError):
        data.apply_style("X", "hungarian")
    with pytest.raises(ValueError):
        data.apply_style("X", "pirate")


# --------------------------------------------------------------------------- assembly / prevalence

def _prompts(n: int) -> list[dict]:
    return [data._record("refusal", "advbench" if i % 2 else "alpaca", str(i), f"prompt number {i} about topic {i * 7}",
                         "harmful" if i % 2 else "harmless") for i in range(n)]


def _generations(prompts: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({"prompt_id": [p["prompt_id"] for p in prompts], "formatted_prompt": "f", "prompt_n_tokens": 5,
                         "response": "r", "finish_reason": "eos", "response_n_tokens": 3})


def _labels(prompts: list[dict], seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"prompt_id": [p["prompt_id"] for p in prompts], "label": rng.randint(0, 2, len(prompts)),
                         "label_strict": None, "ambiguous": rng.rand(len(prompts)) < 0.2, "evidence": "e"})


def test_prompt_record_ids_and_groups():
    a = data._record("refusal", "advbench", "12", "text", "harmful")
    b = data._record("refusal", "advbench", "12", "styled text", "harmful", style="terse")
    assert len(a["prompt_id"]) == 16 and a["prompt_id"] != b["prompt_id"]
    assert a["group_id"] == b["group_id"] == "advbench:12"  # style variants share the leakage group
    assert data._record("refusal", "alpaca", "12", "text", "harmless")["prompt_id"] != a["prompt_id"]


def test_assemble_joins_and_reports_missing_ids():
    prompts = _prompts(30)
    df = data.assemble(prompts, _generations(prompts), _labels(prompts))
    assert len(df) == 30 and df["prompt_id"].is_unique
    for column in ("text", "group_id", "response", "finish_reason", "label", "ambiguous", "evidence", "meta"):
        assert column in df.columns
    assert df["label"].dtype == np.int64 and df["ambiguous"].dtype == bool
    assert isinstance(df["meta"].iloc[0], str)
    with pytest.raises(KeyError, match="missing from generations"):
        data.assemble(prompts, _generations(prompts[:-3]), _labels(prompts))
    with pytest.raises(KeyError, match="missing from labels"):
        data.assemble(prompts, _generations(prompts), _labels(prompts[:-3]))


def test_prevalence_table():
    df = pd.DataFrame({"label": [1, 1, 0, 0, 0, 0], "source": list("aabbbb"), "prompt_category": list("xxxyyy"),
                       "style": ["plain"] * 6})
    table = data.prevalence(df)
    overall = table[table["by"] == "overall"].iloc[0]
    assert (overall["n"], overall["n_pos"]) == (6, 2) and overall["pos_rate"] == pytest.approx(1 / 3)
    by_source = table[table["by"] == "source"].set_index("value")["pos_rate"].to_dict()
    assert by_source == {"a": 1.0, "b": 0.0}


def test_build_dataset_end_to_end(tiny_cfg):
    from src.utils import paths, read_parquet

    prompts = _prompts(700)
    main = data.build_dataset(tiny_cfg, "refusal", prompts, _generations(prompts), _labels(prompts))
    p = paths(tiny_cfg, "refusal")
    assert p.dataset.exists() and p.unused.exists() and (p.dataset_dir / "prevalence.csv").exists()
    assert len(main) == 400 and main["label"].mean() == pytest.approx(0.5)
    assert not main["ambiguous"].any()
    data.check_splits(main)
    unused = read_parquet(p.unused)
    assert not set(unused["prompt_id"]) & set(main["prompt_id"])
    assert not set(unused["group_id"]) & set(main.loc[main["split"] == "train", "group_id"])
    assert read_parquet(p.dataset)["prompt_id"].tolist() == main["prompt_id"].tolist()
    assert data.load_dataset_df(tiny_cfg, "refusal")["split"].tolist() == main["split"].tolist()


# --------------------------------------------------------------------------- manual validation

def _labeled(n: int = 300) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "prompt_id": [f"p{i:04d}" for i in range(n)], "text": [f"prompt {i}" for i in range(n)],
        "response": [f"response {i}" for i in range(n)], "label": rng.randint(0, 2, n),
        "ambiguous": rng.rand(n) < 0.1, "evidence": "e",
    })


def test_export_validation_sample_composition(tmp_path):
    df = _labeled()
    data.export_validation_sample(df, tmp_path, n=50, seed=0)
    annotate, key = pd.read_csv(tmp_path / "annotate.csv"), pd.read_csv(tmp_path / "key.csv")
    assert list(annotate.columns) == ["sample_id", "prompt_text", "response", "human_label"]  # no automatic label
    assert list(key.columns) == ["sample_id", "prompt_id", "auto_label", "ambiguous", "evidence"]
    assert len(annotate) == len(key) == 50 and annotate["human_label"].isna().all()
    assert annotate["sample_id"].tolist() == key["sample_id"].tolist()
    assert int(key["ambiguous"].sum()) == 10
    clear = key[~key["ambiguous"]]
    assert int((clear["auto_label"] == 1).sum()) == 20 and int((clear["auto_label"] == 0).sum()) == 20


def test_export_validation_sample_splits_the_shortfall(tmp_path):
    df = _labeled()
    df["ambiguous"] = False
    df.loc[df.index[:4], "ambiguous"] = True  # only 4 ambiguous rows: 6 missing → 3 extra per class
    data.export_validation_sample(df, tmp_path, n=50, seed=0)
    key = pd.read_csv(tmp_path / "key.csv")
    clear = key[~key["ambiguous"]]
    assert int(key["ambiguous"].sum()) == 4
    assert int((clear["auto_label"] == 1).sum()) == 23 and int((clear["auto_label"] == 0).sum()) == 23


def test_score_validation(tmp_path):
    data.export_validation_sample(_labeled(), tmp_path, n=50, seed=0)
    with pytest.raises(ValueError):  # nothing annotated yet
        data.score_validation(tmp_path)

    key = pd.read_csv(tmp_path / "key.csv")
    annotate = pd.read_csv(tmp_path / "annotate.csv")
    human = key["auto_label"].astype(str).tolist()
    human[0] = "x"  # unclear
    flipped = 1 if human[1] == "0" else 0
    human[1] = str(flipped)  # one disagreement
    annotate["human_label"] = human
    annotate.to_csv(tmp_path / "annotate.csv", index=False)

    report = data.score_validation(tmp_path)
    assert report["n"] == 49 and report["n_unclear"] == 1
    assert report["agreement"] == pytest.approx(48 / 49)
    assert 0.9 < report["kappa"] < 1.0
    assert len(report["disagreements"]) == 1 and report["disagreements"][0]["sample_id"] == key["sample_id"].iloc[1]
    cm = report["confusion_matrix"]
    assert cm["tp"] + cm["tn"] == 48 and cm["fp"] + cm["fn"] == 1
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["n"] == 49


def test_normalize_text_used_for_dedup():
    assert normalize_text("I’m  Sorry") == "i'm sorry"
