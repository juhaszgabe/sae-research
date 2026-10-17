"""Platform pretest (spec §17): PASS / WARN / FAIL per check, written to pretest/{model}/report.json.

Exits non-zero on any FAIL. Run on gemma-3-270m-it first, then on the main model.
"""

import random
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from _common import base_parser, setup

from src import activations, data, labeling
from src import model as model_lib
from src import sae as sae_lib
from src.utils import derive_seed, get_logger, paths, save_run_info, write_json

log = get_logger("pretest")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

PROBE_TEXTS = [
    "What is the capital of France?",
    "Write a short poem about the sea.",
    "Explain how a bicycle stays upright.",
    "Give me three tips for learning a new language.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How do vaccines train the immune system?",
    "List the planets of the solar system in order.",
    "What is the difference between weather and climate?",
]
BUILDERS = {
    "refusal": data.build_refusal_prompts,
    "format_break": data.build_format_prompts,
    "hedging": data.build_hedging_prompts,
}


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for piece in v.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_versions(ctx: dict) -> tuple[str, dict]:
    detail = {"sae_lens": version("sae-lens"), "transformers": version("transformers"), "torch": version("torch")}
    ok = _version_tuple(detail["sae_lens"]) >= (6, 51) and _version_tuple(detail["transformers"]) < (6,)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        detail.update(gpu=torch.cuda.get_device_name(0), gpu_free_gb=free / 2**30, gpu_total_gb=total / 2**30)
    else:
        detail["gpu"] = None
    return (PASS if ok else FAIL), detail


def check_load_model(ctx: dict) -> tuple[str, dict]:
    bundle = model_lib.load_model(ctx["cfg"])  # asserts layer count, d_model and special ids
    ctx["bundle"] = bundle
    return PASS, {"hf_id": bundle.spec.hf_id, "revision": bundle.revision, "n_layers": len(bundle.layers),
                  "d_model": bundle.spec.d_model, "device": str(bundle.device)}


def check_tokenizer(ctx: dict) -> tuple[str, dict]:
    bundle = ctx["bundle"]
    formatted = [model_lib.format_prompt(bundle.tokenizer, t) for t in PROBE_TEXTS[:5]]
    enc = model_lib.tokenize(bundle, formatted, max_len=ctx["cfg"].get("extraction.max_prompt_tokens"))
    positions = [model_lib.find_positions(enc["input_ids"][i], enc["attention_mask"][i], bundle.ids)
                 for i in range(len(formatted))]
    ok = bundle.ids == model_lib.EXPECTED_IDS
    return (PASS if ok else FAIL), {"tokenizer_class": type(bundle.tokenizer).__name__, "ids": bundle.ids,
                                    "example_positions": positions[0]}


def check_hooks(ctx: dict) -> tuple[str, dict]:
    res = activations.check_hooks_vs_hidden_states(ctx["bundle"], PROBE_TEXTS)
    return (PASS if res["ok"] else FAIL), res


def check_padding(ctx: dict) -> tuple[str, dict]:
    bundle = ctx["bundle"]
    res = activations.check_padding_equivalence(bundle, PROBE_TEXTS, bundle.spec.n_layers // 2)
    return (PASS if res["ok"] else FAIL), res


def check_generation(ctx: dict) -> tuple[str, dict]:
    res = activations.check_generation_equivalence(ctx["bundle"], PROBE_TEXTS)
    return (PASS if res["ok"] else WARN), res


def _last_acts(ctx: dict, layers: list[int]) -> dict[int, np.ndarray]:
    """`last` activations of 64 prompts (the eight probe texts with varied prefixes)."""
    if "acts" not in ctx:
        texts = [f"{prefix} {t}" for prefix in ("", "Please answer:", "Quick question.", "Hi!", "In one paragraph:",
                                                "Be concise.", "For a school project:", "I was wondering:")
                 for t in PROBE_TEXTS]
        out = activations.extract(ctx["bundle"], [t.strip() for t in texts], layers, positions=("last",), pooled=(),
                                  batch_size=ctx["cfg"].get("extraction.batch_size"))
        ctx["acts"] = {layer: out[layer]["last"] for layer in layers}
    return ctx["acts"]


def check_sae_backends(ctx: dict) -> tuple[str, dict]:
    cfg = ctx["cfg"]
    layers = list(cfg.model.layers_sae)
    acts = _last_acts(ctx, layers)
    detail, ok = {}, True
    for layer in layers:
        x = torch.from_numpy(acts[layer])
        a = sae_lib.load_sae(cfg, layer, "16k", device="cpu", backend="saelens")  # raises on PT↔IT mismatch (R5)
        b = sae_lib.load_sae(cfg, layer, "16k", device="cpu", backend="raw")
        diff = float((a.encode(x) - b.encode(x)).abs().max())
        detail[f"layer_{layer}"] = {"max_abs_encode_diff": diff, "model": a.info.model_hf_id, "sae_id": a.info.sae_id}
        ok &= diff <= 1e-5
        sae_lib.unload(a)
        sae_lib.unload(b)
    return (PASS if ok else FAIL), detail


def _pt_sae(cfg, layer: int, width: str, l0: str) -> sae_lib.JumpReLUSAE:
    """The PT SAE with the same id, loaded raw (load_sae refuses it on an IT model by design)."""
    from huggingface_hub import hf_hub_download
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    from safetensors.torch import load_file

    entry = get_pretrained_saes_directory()[cfg.model.sae_release.replace("-it-", "-pt-")]
    folder = entry.saes_map[f"layer_{layer}_width_{width}_l0_{l0}"]
    p = load_file(hf_hub_download(entry.repo_id, "params.safetensors", subfolder=folder))
    return sae_lib.JumpReLUSAE(p["w_enc"], p["b_enc"], p["w_dec"], p["b_dec"], p["threshold"])


def check_sae_quality(ctx: dict) -> tuple[str, dict]:
    cfg = ctx["cfg"]
    layers, l0 = list(cfg.model.layers_sae), cfg.get("sae.l0")
    acts = _last_acts(ctx, layers)
    detail, statuses = {}, []
    for layer in layers:
        sae = sae_lib.load_sae(cfg, layer, "16k", l0, device="cpu")
        q = sae_lib.sae_quality(sae, acts[layer])
        status = sae_lib.quality_status(q, sae.info.l0_target)
        statuses.append(status)
        detail[f"layer_{layer}"] = {**q, "l0_target": sae.info.l0_target, "status": status}
        sae_lib.unload(sae)
    middle = layers[len(layers) // 2]
    try:  # informational: the PT SAE on IT activations is expected to reconstruct worse
        pt = _pt_sae(cfg, middle, "16k", l0)
        detail["pt_sae_on_it_activations"] = {"layer": middle, **sae_lib.sae_quality(pt, acts[middle])}
    except Exception as e:
        detail["pt_sae_on_it_activations"] = {"layer": middle, "error": f"{type(e).__name__}: {e}"}
    status = FAIL if "error" in statuses else WARN if "warning" in statuses else PASS
    return status, detail


def check_neuronpedia(ctx: dict) -> tuple[str, dict]:
    cfg = ctx["cfg"]
    infos = [sae_lib.lookup_sae(cfg, layer, "16k", "medium") for layer in cfg.model.layers_sae]
    ids = {f"layer_{i.layer}": i.neuronpedia_id for i in infos}
    fetched = sae_lib.fetch_neuronpedia(infos[0], 0) if infos[0].neuronpedia_id else None
    ok = all(ids.values()) and fetched is not None
    return (PASS if ok else WARN), {"neuronpedia_ids": ids, "example_url": sae_lib.neuronpedia_url(infos[0], 0),
                                    "fetch_ok": fetched is not None,
                                    "example_explanations": (fetched or {}).get("explanations", [])[:2]}


def check_prevalence(ctx: dict) -> tuple[str, dict]:
    cfg, bundle, n = ctx["cfg"], ctx["bundle"], ctx["n_prevalence"]
    detail, ok = {}, True
    tokens = seconds = prompts_done = 0
    for behavior in ctx["behaviors"]:
        prompts = BUILDERS[behavior](cfg)
        prompts = random.Random(derive_seed(cfg["seed"], "pretest", behavior)).sample(prompts, min(n, len(prompts)))
        t0 = time.perf_counter()
        gens = model_lib.generate(bundle, [p["text"] for p in prompts], cfg.get(f"generation.max_new_tokens.{behavior}"),
                                  cfg.get("generation.batch_size"), show_progress=False)
        seconds += time.perf_counter() - t0
        tokens += sum(g["response_n_tokens"] for g in gens)
        prompts_done += len(prompts)
        rows = pd.DataFrame([{"prompt_id": p["prompt_id"], "json_schema": p["json_schema"], **g}
                             for p, g in zip(prompts, gens)])
        labels = labeling.label_frame(behavior, rows)
        rate = float(labels["label"].mean())
        detail[behavior] = {"n": len(labels), "positive_rate": rate, "ambiguous_rate": float(labels["ambiguous"].mean()),
                            "truncated_rate": float(np.mean([g["finish_reason"] == "length" for g in gens]))}
        ok &= 0.05 <= rate <= 0.95
    ctx["throughput"] = {"tokens": tokens, "seconds": seconds, "prompts": prompts_done}
    return (PASS if ok else WARN), detail


def check_throughput(ctx: dict) -> tuple[str, dict]:
    cfg, t = ctx["cfg"], ctx.get("throughput")
    if not t or not t["seconds"]:
        return WARN, {"error": "no generation timing available (prevalence check did not run)"}
    prompts_per_s = t["prompts"] / t["seconds"]
    pool_sizes = {}
    for behavior in ctx["behaviors"]:
        if behavior == "refusal":
            specs = cfg["refusal_sources"]
            pool_sizes[behavior] = sum(specs[s].get("max_n") or 500 for s in cfg["refusal_main_sources"])
        elif behavior == "format_break":
            pool_sizes[behavior] = cfg["format_candidates"]
        else:
            pool_sizes[behavior] = 700
    return PASS, {
        "tokens_per_s": t["tokens"] / t["seconds"], "prompts_per_s": prompts_per_s,
        "candidate_pool_sizes": pool_sizes, "dataset_target_n": cfg.get("dataset.target_n"),
        "projected_generation_minutes": {b: n / prompts_per_s / 60 for b, n in pool_sizes.items()},
    }


CHECKS = [
    ("versions", check_versions, False),
    ("load_model", check_load_model, False),
    ("tokenizer_positions", check_tokenizer, True),
    ("hooks_vs_hidden_states", check_hooks, True),
    ("padding_equivalence", check_padding, True),
    ("generation_equivalence", check_generation, True),
    ("sae_backends", check_sae_backends, True),
    ("sae_quality", check_sae_quality, True),
    ("neuronpedia", check_neuronpedia, False),
    ("prevalence", check_prevalence, True),
    ("throughput", check_throughput, False),
]


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-prevalence", type=int, default=50, help="prompts per behavior for the prevalence check")
    args = parser.parse_args()
    cfg, behaviors = setup(args)
    out_dir = paths(cfg, behaviors[0]).pretest_dir
    save_run_info(cfg, out_dir, "pretest")

    ctx = {"cfg": cfg, "behaviors": behaviors, "n_prevalence": args.n_prevalence}
    report = {}
    for name, check, needs_model in CHECKS:
        if needs_model and "bundle" not in ctx:
            report[name] = {"status": FAIL, "detail": {"error": "model not loaded"}}
        else:
            try:
                status, detail = check(ctx)
            except Exception as e:  # a failing check must not hide the others
                status, detail = FAIL, {"error": f"{type(e).__name__}: {e}"}
            report[name] = {"status": status, "detail": detail}
        log.info("%-24s %s", name, report[name]["status"])
        write_json(out_dir / "report.json", {"model": cfg.model.name, "checks": report})

    if "bundle" in ctx:
        model_lib.free_model(ctx["bundle"])
    failed = [name for name, r in report.items() if r["status"] == FAIL]
    log.info("pretest %s; report: %s", f"FAILED ({', '.join(failed)})" if failed else "passed", out_dir / "report.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
