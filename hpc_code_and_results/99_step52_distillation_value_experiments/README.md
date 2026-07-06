# step52_distillation_value_experiments/

## What question this folder answers

**Not** "does the teacher beat MTGS / a fully-supervised specialist?" — that
was never the question (see 2026-07-05 conversation: MTGS is fully-
supervised, fine-tuned directly on VAT, and uses temporal context even for
its "static" number — not comparable to a zero-shot pipeline).

**Actually answers Objective 2, verbatim**: *"Determine whether a VLM can
generate social gaze training labels that are able to produce equivalent or
superior student performance vs raw ground-truth annotations."* The number
that matters is teacher-trained student ADE vs GT-trained student ADE,
evaluated against real GT — not the teacher's raw ADE in isolation.

## Why this folder exists (context if you've forgotten)

The original interim-report test of this (Step5b, April) used only 85
frames — indicative, not decisive. On 2026-07-05, a raw-ADE comparison
against MTGS (published SOTA) triggered a "our approach might be broken"
scare. Investigation showed the comparison was invalid (different task:
zero-shot vs fully-supervised; different population; different metric
protocol). These three scripts re-run the ACTUAL decisive test — teacher
vs GT student performance — at full production scale instead of 85 frames,
using the final frozen teacher labels (`labels_hybrid_targeted.jsonl`,
InternVL3 + GDINO targeted-phrase prompt).

## Files, in the order you should think about them

1. **`step52_distillation_full_scale.py`** — the core test. Trains two
   ViT-Tiny students (one on teacher `pred_x/pred_y`, one on `gt_x/gt_y`) on
   identical held-out clips, evaluates both against real GT. Reports ONE
   pooled ADE per model across all on-screen frames (social + object mixed).

2. **`step52b_distillation_by_gaze_type.py`** — same core test, but also
   breaks the final ADE down by `gaze_type` (social vs object) for both
   models. Answers "if there's a gap, is it in social, object, or both?" —
   which the pooled script can't tell you. Also fixes a split-determinism
   bug present in the pooled script (see GOTCHAS below) — was fixed here
   first, not backported to the pooled script, so don't assume both scripts'
   internal train/val splits are identical or individually reproducible.

3. **`check_shortcut_collapse.py`** — run AFTER 1 and 2 finish, against
   their saved checkpoints. Does NOT retrain anything. Checks whether a
   checkpoint's ADE reflects genuine per-frame spatial grounding or a
   collapsed per-show positional-average shortcut — the exact failure mode
   that made the LoRA fine-tuning attempt (`step48`, same evening) LOOK like
   a 0.235 ADE improvement when it was actually memorizing per-show
   constants. Run once per checkpoint (4 runs: teacher/gt x pooled/by-type).
   **Don't trust either distillation script's headline ADE until this has
   been run on it.**

## How to run

```
sbatch distillation_full_targeted.sub       # step52
sbatch distillation_by_gaze_type.sub        # step52b
# wait for both to finish, then, per checkpoint:
sbatch shortcut_check.sub                   # edit CHECKPOINT path inside first
```

## Gotchas / things that already bit us once

- **`DISTILL_LABELS_PATH` must be set explicitly** in the `.sub` file (both
  scripts default to a bare relative filename that won't exist in the
  working directory otherwise — this caused an instant FileNotFoundError
  crash the first time). Confirm the actual path with `find` before
  trusting any hardcoded value here, paths have moved before.
- **Off-screen filtering**: must use `gt_px_x`/`gt_px_y == -1` (raw pixel
  sentinel), NOT a `None`-check on `gt_x`/`gt_y`. Off-screen rows have
  `gt_x`/`gt_y` populated with mis-normalized garbage (`-1/img_width`), not
  `None` — a `None`-check silently lets them corrupt the on-screen ADE.
  Both scripts already have this fix; if you ever copy this pattern
  elsewhere, carry the fix with it.
- **Split reproducibility**: `step52`'s train/val clip split uses
  `list(set(...))` before shuffling, which is NOT reproducible across
  separate process runs (Python's hash randomization). `step52b` fixed this
  via `sorted(set(...))`. Don't assume you can reconstruct `step52`'s exact
  val set from outside that one run.

## Where the actual numbers end up

- `experiment_results_full_targeted/` — step52's output (pooled ADE,
  checkpoints, comparison.json, plot)
- `experiment_results_full_targeted_by_gaze_type/` — step52b's output (same,
  plus per-gaze-type breakdown in the summary text)

## What a good outcome looks like, and what to do next

See conversation notes 2026-07-05/06 for the full decision tree, but in
short: if teacher-trained ADE tracks GT-trained ADE closely (both pooled
and per gaze-type) — Objective 2 is answered, freeze
`labels_hybrid_targeted.jsonl` as final, move to temporal work. If object
specifically shows a real gap — the head-box/subject-marking fix
(discussed 2026-07-04, not yet built) is the next well-motivated thing to
try, not a reason to reconsider the whole approach.
