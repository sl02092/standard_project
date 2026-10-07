# Distilling Structured Social Attention for Proactive Gaze Anticipation in Human-Robot Interaction

Codebase for my MSc Artificial Intelligence dissertation of the same name (University of Surrey, 2026), supervised by Dr. Di Fu.

---

## What this project does

A social robot that can predict where a person is *about to* look gains a short but useful head start. It can turn towards the target, hand over an object, or take its turn in a conversation in time to be useful. To do that on a robot, the model has to run in real time on modest hardware. That rules out the large vision-language models that are best at understanding social scenes.

This project asks two questions:

- Can the social-attention knowledge in those large models be **distilled into a small student network**?
- Does that distilled knowledge support **anticipation**, rather than just reaction?

The work has two stages:

1. **Static gaze estimation.** A compact ViT student is trained on pseudo-labels from four different label sources. Everything else stays fixed, so the effect of the *label source* on student quality can be measured on its own.
2. **Temporal anticipation.** A lightweight sequence head is trained on the frozen student's features. It predicts gaze 100 ms, 300 ms and 500 ms into the future. It is measured against a deliberately hard baseline, which copies the last known gaze position forward.

The second stage needs that baseline. Gaze changes slowly from frame to frame, so any sequence model will look reasonable on this task. The real test is whether it beats doing nothing at all.

---

## Results at a glance

<!-- Add two or three headline findings here, copied from the submitted dissertation (v1.7). -->

**Deployment cost** (CPU, from `004_benchmark_inference/benchmark_cpu.json`):

| Component | Parameters | Latency (ms) |
|---|---|---|
| Static student, ViT-Tiny | 5.7 M | 13.9 |
| Static student, ViT-Small | 21.9 M | 35.2 |
| Temporal head, GRU | 0.11 M | 0.21 |
| Temporal head, Transformer | 0.28 M | 0.32 |

---

## The four label sources

The central experiment trains the same student, with the same training procedure, on four different label sources:

| Condition | Label source |
|---|---|
| `gt` | Human ground-truth annotations (the upper reference) |
| `mtgs` | MTGS, an existing trained gaze model, used as the teacher |
| `teacher_hybrid` | InternVL3-8B for social targets; Grounding DINO localises object targets |
| `teacher_internvl3_alone` | InternVL3-8B alone, with no detector |

Comparing these shows how much of a student's quality comes from the labels themselves. Model size and training budget are held constant.

---

## Datasets

| Dataset | Role |
|---|---|
| **VideoAttentionTarget (VAT)** | Primary in-domain dataset |
| **ChildPlay** | In-domain; child-directed social attention |
| **VACATION** | In-domain; object- and person-directed attention |
| **UCO-LAEO** | Held out from static training; used for transfer evaluation |

The static students are trained jointly on VAT, ChildPlay and VACATION.

The raw datasets are not included here. They must be obtained from their original sources.

---

## Repository layout

The repository is a set of pipeline stages, not a single installable package. Each folder's numeric prefix shows which stage it belongs to.

### `000_*` — Dataset preparation and per-dataset experiments
`000_ChildPlay`, `000_UCO-LAEO`, `000_VACATION`

This stage parses each dataset, extracts the frames and normalises the annotations. The datasets use different annotation formats, so each one has its own preparation code. The output is a uniform per-frame manifest containing:
- the head bounding box
- the gaze target
- the gaze type (social, object or off-screen)
- the split

These folders also hold each dataset's temporal window builders, last-known-position baselines and proof-of-concept temporal training scripts (`train_temporal_poc_*.py`).

VAT's preparation and checking scripts are in `002_teacher_pipeline_vat` and `001_teacher_pipeline_library_temporal/vat`.

### `001_*` — Temporal window construction
`001_teacher_pipeline_library_temporal`

This stage builds the sliding-window manifests that the anticipation stage uses. Each window has eight consecutive context frames, a stride of four, and prediction targets at three horizons. It also includes scripts that check frame contiguity and horizon offsets against the real data.

### `002_*` — Teacher pipelines
`002_teacher_pipeline_vat`, `002_teacher_pipeline_childplay`, `002_teacher_pipeline_vacation`, `002_teacher_pipeline_uco-laeo`

These folders generate pseudo-labels for each dataset under each teacher configuration:
- `step40_*` — InternVL3-8B alone
- `step46_*` — the InternVL3-8B + Grounding DINO hybrid
- `MTGS*` / `export_*_pseudolabels.py` — MTGS pseudo-label export

`002_teacher_pipeline_internv3+hybrid_fix` holds the follow-up experiments proposed in §7.4 of the dissertation. These are **not** the labels behind the submitted results. There are two variants:
- **`_noleak`**: removes the target person's face centre from the InternVL3 prompt. The original prompt returned that supplied coordinate unchanged on 88.1% of social frames (§3.2.3).
- **`_alwaysdino`**: always takes Grounding DINO's best detection and never falls back to ground truth. Every hybrid label then comes from a single process.

`patch_teacher_noleak.py` generates both variants from the frozen originals. It never modifies the originals.

### `003_*` — Distillation
`003_distillation_static`, `003_distillation_temporal`

`003_distillation_static` holds the student model (`gaze_student_model.py`), the training script (`distillation_static_vit_small/distillation_v1.py`), and the results for ViT-Tiny and ViT-Small. It also holds a confidence-threshold sweep for the hybrid teacher, and the UCO-LAEO transfer evaluation.

`003_distillation_temporal` builds temporal windows from the static test splits. It extracts frozen per-frame student features, then evaluates the anticipation heads against the copy-forward baseline. The heads are GRU and Transformer, each with a residual variant. Significance is tested with a paired bootstrap.

### `004_*` — Analysis and evaluation
| Folder | Purpose |
|---|---|
| `004_benchmark_inference` | Latency and throughput on CPU and GPU |
| `004_compute_trivial_baselines` | Trivial baselines that any real model must beat |
| `004_gaze_lle` | Comparison with Gaze-LLE, an external published gaze model |
| `004_intersection_comparison` | Comparisons across conditions on matched frames |
| `004_vi-lam_heatmaps` | The ViLAD extension: auxiliary heatmap supervision, run for each label source, with bootstrap confidence intervals |
| `004_predicted_point_overlays`, `004_temporal_overlay_figures` | Qualitative figures |

---

## Method summary

**Student.** A ViT backbone, in Tiny and Small sizes. It takes a scene image and a head crop, and predicts a normalised gaze point plus the probability that the target is in the frame. The coordinate loss is applied only to frames where the target is on screen, because off-screen targets have no meaningful coordinate.

**Anticipation head.** Per-frame student embeddings are cached once. A small sequence model (0.11–0.28 M parameters) then maps an eight-frame window to a future gaze point. The residual variants predict an *offset* from the last known position, not an absolute coordinate. They are initialised so that the model starts exactly at the copy-forward baseline and has to earn any improvement on it.

**Metrics.**
- Average Displacement Error (ADE), normalised by each clip's true pixel dimensions.
- PCK at 0.05, 0.10 and 0.15.
- ROC-AUC for the in-frame/out-of-frame head.

Significance is tested against the baseline with a paired bootstrap (10,000 resamples, 95% confidence interval). Results are reported separately for each gaze type, not pooled.

**Evaluation discipline.** Results are reported on held-out test populations built from each dataset's official test split. Matched-subset comparisons make sure the model and the baseline are always scored on the same frames. MTGS was pretrained on VAT's training split, so VAT is evaluated on a test-shows-only population (the `*_testshows` teacher scripts).

---

## Reproducing

The pipelines were run on a SLURM cluster. The `.sub` job files next to the scripts show the exact invocation used, along with the expected paths and arguments. Most scripts also explain their usage in the module docstring.

```
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.11.0+cu130`. Install PyTorch for your own CUDA version first, from pytorch.org, then install the rest.

Broad order of execution:

```
000_*   prepare datasets              →  per-frame manifests
002_*   run teacher pipelines         →  pseudo-label sets
003_*   distil the static student     →  student checkpoints
001_*   build temporal windows        →  windowed manifests
003_*   extract features, evaluate the anticipation heads
004_*   analysis, baselines and figures
```

Model checkpoints and cached features are not included, because of their size.

---

## Notes

This is a research codebase, not a library. The scripts are meant to be run one at a time, with explicit configuration. Many of them contain detailed comments recording bugs that were found and fixed during development. Those comments are deliberate. They are often the clearest explanation of why a piece of logic is written the way it is.

---

## Citation

```
Lewis, S. (2026). Distilling Structured Social Attention for Proactive Gaze
Anticipation in Human-Robot Interaction. MSc dissertation, University of Surrey.
```
