# VAT Temporal Window-Builder — Findings Reference

**Status:** VAT temporal manifest built and validated. ChildPlay/VACATION not
yet started. Written as a reference for the dissertation write-up and as a
starting point for further exploration later.

**Last updated:** 2026-07-06

---

## What this covers

Objective 4 (proactive temporal anticipation) needs contiguous multi-frame
context windows, not individually-sampled frames — a different requirement
from the static distillation pipeline (Objectives 2/3). This work builds
the temporal manifest for VAT specifically, investigates a data-quality
question that came up while doing it, and settles that question with
evidence rather than assumption.

**Scripts (in `/mnt/user-data/outputs/`, or wherever you've since moved them):**
- `step_temporal_window_builder_vat.py` — the window-builder itself.
- `inspect_gap_report.py` — small utility to pull exact missing frame
  filenames for a given show out of the gap report, for manual spot-checks.

**Outputs it produces (paths as written by the script, run from wherever
you invoke it):**
- `vat_temporal_manifest.jsonl` — the actual temporal manifest, per the
  finalized schema (see `temporal_manifest_schema.md`, 2026-07-06).
- `vat_temporal_gap_report.csv` — every mid-window annotation gap found,
  logged per (show, clip, subject, window).
- `vat_temporal_gap_blocks.csv` — the same gaps collapsed into true
  contiguous runs (one row per real gap, not one row per overlapping
  window that happens to touch it).

---

## Why this was built this way

Step30_clip_scanner_2.py (the static pipeline's clip scanner) was **not**
reused wholesale. Its clip-quality filter (`should_include` — min subjects,
min frames, social %, off-screen %) was kept, because it's the same
quality bar the static pipeline already uses. Its frame-sampling
(`FRAME_SAMPLE_EVERY_N`) and per-show frame cap (`MAX_FRAMES_PER_SHOW`)
were deliberately **dropped** — both would break the frame-to-frame
contiguity a temporal window depends on (sampling skips frames; the cap
can truncate a clip mid-stream).

**Design decisions carried over from the schema discussion (2026-07-06):**
- Window length: 8 context frames, stride 4.
- Horizons: 100ms / 300ms / 500ms (continuity with the original interim-
  review plan), computed as frame-offsets — 2/7/12 frames at VAT's 24fps
  nominal assumption (VAT has no authoritative fps; this is stated and
  caveated, not measured — see schema doc for the full reasoning).
- One combined manifest (not three per-horizon files), filtered by horizon
  at training time.
- Point-only training target; no `target_box` field (kept as per-dataset
  diagnostic elsewhere, not in this manifest).
- **No fallback labels, ever.** A window/horizon is either fully genuine
  or explicitly marked invalid with a reason — never padded, interpolated,
  or substituted.

---

## The gap investigation — what was found, and why it matters

### The numbers (real run, 214 qualifying clips, 1 show dropped for being
under `MIN_FRAMES_PER_SHOW`)

```
Candidate windows (all subjects)     : 21,397
Windows with a mid-window gap        : 644 (3.0%)
Total rows in manifest (windows × 3 horizons) : 64,191
Valid rows                           : 59,138
Invalid rows                         : 5,053
```

Of the 5,053 invalid rows, roughly 1,932 trace back to the 644 gap-
affected windows (× 3 horizons each); the remainder come from horizon
targets falling beyond a clip's end — expected, and worse for the longer
(500ms/12-frame) horizon than the shorter ones, since less clip remains
after the context window to reach a distant target.

### What the gaps actually are

Collapsing the 644 window-level gap flags down to their true underlying
events gives **46 distinct contiguous gap blocks** across all 214 clips —
i.e. this is a rare phenomenon overall, concentrated in a small number of
specific clips, not a pervasive noise floor across the dataset.

**The key finding: 44 of 46 blocks (95.7%) touch the start or end of their
clip** (33 at the start, 11 at the end) — this holds across the full
length range, from 2-frame blocks up to 186-frame blocks. Only **2 blocks
are genuinely mid-clip**, and both are in the same clip (see below).

### Case study: Sound of Music, clip `0_270`

Manual inspection (watching the actual footage, not just the annotation
files) confirmed the mechanism directly: a character is in frame for the
entire clip, but their face is partially hidden (under a duvet) and their
eyes appear closed. VAT's annotators did not assign this subject a gaze
label from frame 4 through frame 126 — then began annotating at frame 127,
which is when the character starts visibly moving and their gaze becomes
determinable.

**Conclusion drawn from this:** VAT's per-subject annotation gaps are not
tracking failures or occlusion noise. They reflect a deliberate annotation
convention — a subject is left unannotated during a sustained span where
their gaze direction genuinely isn't determinable, rather than being
guessed at. This directly validates the "invalidate, never pad" policy:
there is no ground truth to interpolate during these spans, because a
sighted human annotator with full video context declined to provide one.

### The one exception: clip `1438_2248` (also Sound of Music)

The only 2 mid-clip blocks in the entire dataset both fall in this one
clip: subject s01 (frames 1835–1912, 49–58% through the clip) and subject
s02 (frames 1812–1850, 46–51% through — an overlapping but not identical
span, different character). **Not yet manually verified** — worth a
same-method spot-check (pull the exact frame range via
`inspect_gap_report.py`, watch that stretch) before assuming it's the same
phenomenon. Two different characters both losing determinable gaze around
the same mid-clip moment could mean a genuine scene event (e.g. a shared
moment of distraction/darkness/off-camera action) rather than the
individual "not yet gazing" pattern seen in `0_270` — plausible, but
unconfirmed.

### Suggested dissertation framing

> "Analysis of 46 distinct annotation gaps across 214 qualifying VAT clips
> showed 95.7% occurred at clip boundaries — subjects not yet introduced,
> or exiting before the clip's end — consistent with annotators withholding
> gaze labels during periods judged non-determinable rather than annotating
> through occlusion or ambiguity. Manual inspection of a representative
> case (Sound of Music, clip 0_270) confirmed this directly: the affected
> subject's face was obscured and eyes closed for the unannotated span,
> with annotation resuming at the first frame showing clear gaze-directed
> movement. Only 2 gaps (both within a single clip) occurred mid-sequence
> and were not attributable to clip boundaries."

This turns what could look like an unexplained ~3% data loss into a
characterized, evidenced methodological finding — and gives principled
backing (not just "safest default") for invalidating rather than padding
across gaps.

---

## Open items / for later exploration

- **Verify clip `1438_2248`'s two mid-clip blocks** the same way `0_270`
  was verified — the one loose thread in an otherwise fully-explained
  picture.
- **ChildPlay port** — the gap-detection logic (`scan_gaps`,
  `compute_gap_blocks`) and the Phase 1/Phase 2 split were written to be
  dataset-agnostic; porting to ChildPlay should mainly need a new
  clip/annotation loader (ChildPlay's format, not VAT's `.txt` files) with
  the core gap-analysis functions left untouched. Not yet attempted.
- **Per-horizon invalid-rate breakdown** — flagged as worth checking but
  not yet actually pulled from the manifest: filter `vat_temporal_manifest.jsonl`
  by `horizon_frames` and compare invalid-rate per horizon (expectation:
  12-frame/500ms horizon loses more than 2-frame/100ms — confirming this
  would rule out anything unexpected in the horizon logic itself).
- **VACATION** — still pending dataset access; this whole gap-analysis
  approach (block-length distribution, start/end-vs-mid-clip breakdown)
  is a reasonable template to reapply once that data arrives, given how
  much it revealed here from a single pass.

---

## How to pick this back up later

If you come back to this and want to explore further:
- `vat_temporal_gap_blocks.csv` is the single richest artifact — one row
  per real gap, with length, position-in-clip, and start/end flags. Most
  new questions about "what's actually going on with gaps" should start
  there rather than the raw per-window report.
- `inspect_gap_report.py <report.csv> "<show name>"` pulls exact frame
  filenames for manual video spot-checks, same method used for both Sound
  of Music clips above.
- The window-builder script's docstring explains the Phase 1/Phase 2 split
  and the reasoning behind what was and wasn't reused from Step30 — worth
  rereading before modifying it, so any changes stay consistent with why
  it was built this way.
