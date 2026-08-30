"""
bootstrap_ci_static.py — paired bootstrap confidence intervals on the
static four-way teacher-condition ranking, using per-record distances
(per_record_ade_{condition}.json from eval_static_v1.py's own export).

Paired, not independent-samples: all four conditions evaluate on the
identical underlying population (same records, different model
checkpoints) -- pairing by record position is both more rigorous and
matches the exact bootstrap methodology already used for every temporal
claim in this project (10,000 resamples, 95% CI on the mean paired
difference). This replaces the SD-derivation workaround (SD =
sqrt(RMSE^2 - ADE^2), borrowed from a different model/condition) with a
genuine per-record test -- no borrowed variance assumption needed.

Pure Python, no GPU -- the expensive part (running each model on the
full population) already happened when eval_static_v1.py ran.
"""

import os
import json
import random


def load_per_record(path):
    with open(path) as f:
        data = json.load(f)
    return data["per_record_ade"], data["dataset"], data["gaze_type"]


def paired_bootstrap_ci(errors_a, errors_b, n_resamples=10000, seed=42):
    """95% CI on mean(errors_a - errors_b). Positive means A is worse
    (higher error) than B on average."""
    assert len(errors_a) == len(errors_b), "Populations must be paired -- same length required."
    n = len(errors_a)
    diffs = [a - b for a, b in zip(errors_a, errors_b)]
    observed_mean = sum(diffs) / n

    rng = random.Random(seed)
    resample_means = []
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(resample) / n)
    resample_means.sort()
    ci_lo = resample_means[int(0.025 * n_resamples)]
    ci_hi = resample_means[int(0.975 * n_resamples)]

    if ci_lo > 0:
        verdict = "REAL+ (A significantly worse than B)"
    elif ci_hi < 0:
        verdict = "REAL- (A significantly better than B)"
    else:
        verdict = "n.s."
    return {"mean_diff": observed_mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "verdict": verdict}


def main():
    # Fill in the real per_record_ade_{condition}.json paths once available.
    condition_files = {
        "gt": os.environ.get("PER_RECORD_GT", "per_record_ade_gt.json"),
        "mtgs": os.environ.get("PER_RECORD_MTGS", "per_record_ade_mtgs.json"),
        "teacher_internvl3_alone": os.environ.get("PER_RECORD_IV3", "per_record_ade_teacher_internvl3_alone.json"),
        "teacher_hybrid": os.environ.get("PER_RECORD_HYBRID", "per_record_ade_teacher_hybrid.json"),
    }

    loaded = {}
    for cond, path in condition_files.items():
        if not os.path.isfile(path):
            print(f"SKIPPING {cond} -- file not found: {path}")
            continue
        errors, datasets, gaze_types = load_per_record(path)
        loaded[cond] = {"errors": errors, "dataset": datasets, "gaze_type": gaze_types}
        print(f"Loaded {cond}: n={len(errors)}")

    if len(loaded) < 2:
        print("\nNeed at least 2 conditions loaded to compare. Exiting.")
        return

    # Verify populations are genuinely paired (same size, same record order)
    # before trusting any position-based pairing -- fail loud, not silent.
    sizes = {cond: len(v["errors"]) for cond, v in loaded.items()}
    if len(set(sizes.values())) > 1:
        print(f"\nERROR: population sizes differ across conditions: {sizes}")
        print("Pairing by position is invalid unless every condition evaluated the exact same population.")
        return
    reference_cond = list(loaded.keys())[0]
    for cond in loaded:
        if loaded[cond]["dataset"] != loaded[reference_cond]["dataset"]:
            print(f"\nERROR: record order differs between {cond} and {reference_cond} "
                  f"(dataset labels don't match position-by-position). Pairing is invalid.")
            return
    print(f"\nPopulation check passed: all {len(loaded)} conditions share {sizes[reference_cond]} "
          f"identically-ordered records.\n")

    # Ranking order (best to worst pooled ADE) -- adjust based on real results.
    pooled_ade = {cond: sum(v["errors"]) / len(v["errors"]) for cond, v in loaded.items()}
    ranking = sorted(pooled_ade, key=pooled_ade.get)
    print("Pooled ADE (for reference):")
    for cond in ranking:
        print(f"  {cond:<28} {pooled_ade[cond]:.4f}")

    print(f"\n{'='*70}\nPairwise paired bootstrap CIs -- every adjacent pair in the ranking\n{'='*70}")
    results = {}
    for i in range(len(ranking) - 1):
        a, b = ranking[i + 1], ranking[i]  # a = worse, b = better (adjacent in ranking)
        r = paired_bootstrap_ci(loaded[a]["errors"], loaded[b]["errors"])
        results[f"{a}_vs_{b}"] = r
        print(f"  {a} vs {b}: mean_diff={r['mean_diff']:+.4f}  "
              f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {r['verdict']}")

    out_path = "static_ranking_bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump({"pooled_ade": pooled_ade, "ranking": ranking, "pairwise": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
