"""
bootstrap_ci_temporal.py — paired bootstrap CI for the temporal UCO-LAEO
zero-shot results: model error vs. static-copy-forward baseline error,
one test per (static_condition, architecture, horizon) combination.

Structurally different from bootstrap_ci_static.py's own comparison:
that script ranks FOUR SEPARATE FILES (different conditions) against
each other. This script tests ONE FILE at a time, whose two paired
arrays (model_ade, baseline_ade) come from eval_temporal_v1.py's own
per-row export -- the question here is "does the model beat its own
baseline", not "which of several conditions is best". Reuses the same
underlying paired_bootstrap_ci() function (imported, not copied --
bootstrap_ci_static.py has no module-level env-var requirements, so
importing is safe and keeps one definition rather than two that could
drift apart).

Auto-discovers every per_record_ade_temporal_*.json file in the given
directory, rather than requiring one env var per file -- with up to
4 conditions x 4 architectures x 3 horizons (48 files), hand-listing
them would be impractical. Point this at a directory containing
however many of those files exist (one horizon, all horizons, one
architecture, whatever you have) and it processes all of them in a
single run, producing one summary table plus REAL+/REAL-/n.s. verdicts
matching the exact terminology and 10,000-resample/95%-CI convention
already used for every other temporal claim in this project (Section
3.6.3) -- this closes the specific, disclosed gap where the UCO-LAEO
zero-shot result lacked that same formal test (Section 5.6/6.2).
"""

import os
import re
import sys
import json
import glob

from bootstrap_ci_static import paired_bootstrap_ci

FILENAME_PATTERN = re.compile(
    r"per_record_ade_temporal_(?P<condition>.+?)_(?P<arch>GRU|Transformer|ResidualGRU|ResidualTransformer)_(?P<horizon>\d+)ms\.json"
)


def find_temporal_files(directory):
    # recursive=True + ** so this works whether files sit flat in one
    # folder or nested as {condition}/horizon_{ms}ms/ (this project's own
    # eval_temporal_ucolaeo_vit{tiny,small} layout) -- one call finds
    # everything under the given root regardless of depth.
    paths = sorted(glob.glob(os.path.join(directory, "**", "per_record_ade_temporal_*.json"), recursive=True))
    parsed = []
    for path in paths:
        m = FILENAME_PATTERN.search(os.path.basename(path))
        if not m:
            print(f"  SKIPPING (name doesn't match expected pattern): {path}")
            continue
        parsed.append({"path": path, "condition": m.group("condition"),
                        "arch": m.group("arch"), "horizon": int(m.group("horizon"))})
    return parsed


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMPORAL_PER_RECORD_DIR", ".")
    print(f"Scanning for per_record_ade_temporal_*.json files in: {directory}\n")

    files = find_temporal_files(directory)
    if not files:
        print("No matching files found. Nothing to do.")
        return
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  condition={f['condition']:<28} arch={f['arch']:<20} horizon={f['horizon']}ms")

    print(f"\n{'='*90}\nPaired bootstrap CI: model vs. static-copy-forward baseline, per file\n{'='*90}")
    results = []
    for f in files:
        with open(f["path"]) as fh:
            data = json.load(fh)
        model_ade, baseline_ade = data["model_ade"], data["baseline_ade"]
        if len(model_ade) != len(baseline_ade):
            print(f"  SKIPPING {f['path']} -- model_ade and baseline_ade lengths differ "
                  f"({len(model_ade)} vs {len(baseline_ade)}), not a valid paired population.")
            continue

        # baseline - model: positive means model beats baseline (lower error)
        r = paired_bootstrap_ci(baseline_ade, model_ade)
        # Relabel verdict in terms a reader of Section 5's own REAL+/REAL-/n.s.
        # convention expects: REAL+ here means the model is REAL+ improvement
        # over baseline (matches Section 3.6.3's "positive meaning the model
        # is better" framing exactly), not the raw A-vs-B wording
        # paired_bootstrap_ci() uses internally for the generic case.
        if r["ci_lo"] > 0:
            verdict = "REAL+ (model significantly beats baseline)"
        elif r["ci_hi"] < 0:
            verdict = "REAL- (model significantly worse than baseline)"
        else:
            verdict = "n.s."

        print(f"  {f['condition']:<28} {f['arch']:<20} {f['horizon']:>4}ms  "
              f"improvement={r['mean_diff']:+.4f}  CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {verdict}")
        results.append({**f, "n": len(model_ade), "improvement_mean": r["mean_diff"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "verdict": verdict})

    out_path = os.path.join(directory, "temporal_ucolaeo_bootstrap_ci.json")
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
