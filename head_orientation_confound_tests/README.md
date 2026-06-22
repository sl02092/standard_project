# Head-Orientation Confound Check — Scripts

This folder contains the standalone scripts used to investigate the head-orientation /
gaze-target confound (cf. arXiv 2506.05412) against the project's teacher pipeline.

**None of these scripts are part of the production pipeline.** They do not write to
`labels_test.jsonl`, `labels_full.jsonl`, or any checkpoint/progress file used by
`step40_teacher_pipeline_004.py`. They are read-only against the VAT dataset and exist purely
to test prompt variants and compare model behaviour against ground truth on a small, fixed
set of hand-picked frames.

See `confound_check_findings.md` (or the equivalent Word doc) for what was actually found.
This README only covers what the code does.

---

## Script history (chronological — later scripts supersede earlier ones)

| File | Purpose | Status |
|---|---|---|
| `step43_prompt_compare.py` | First version. A/B test of two InternVL3-8B prompt variants against the original `labels_test.jsonl` sample. | Superseded |
| `step43b_prompt_compare.py` | Adds a 3rd variant (`v6_multicandidate`) requiring a different argument shape; introduces the `PROMPT_ARGS` dispatch pattern so each prompt variant can take different arguments. | Superseded |
| `step43d2_prompt_compare.py` | Rewrite: drops fixed Person A/B framing in favour of dynamic, presence-aware prompts (spatial left/centre/right labels; no fabricated "Person B" for solo frames). Adds direct annotation-file lookup (`resolve_context`) instead of relying on a pre-built manifest, to support clips outside the original CSV (e.g. Project Runway). | Superseded — had two lookup bugs, see below |
| `step43d3_prompt_compare.py` | Bug-fix version of `step43d2`. Fixes two issues found while chasing down a frame-mismatch problem:<br>1. `find_dir_by_clip_id` now requires an **exact** folder-name match (was a substring match, risking wrong-folder collisions like `650_1775` matching inside `1650_1775`).<br>2. `parse_txt_file_for_frame` now requires an **exact** frame-number match (was falling back to a last-two-digits suffix match, which could silently return a completely different frame's data). | **Superseded by step43e — but the lookup fixes here are the ones that matter; they carry forward unchanged into every later script.** |
| `step43e_prompt_compare_qwen.py` | Same as `step43d3`, with off-screen-handling improvements to the three prompt variants (explicit `Is_Off_Screen: Yes/No` forced field, `(OFF, OFF)` token instead of relying on the model to emit `(-1,-1)` unprompted). Still targets InternVL3-8B at this point — the "qwen" in the filename refers to it being prepared as the base for the Qwen conversion, not yet running Qwen. | Superseded |
| `step43e2_prompt_compare_qwen.py` | **First actual Qwen2.5-VL-7B-Instruct run.** Same prompts/lookup/frames as `step43e`, swapped to `Qwen2_5_VLForConditionalGeneration`. | Used for the Qwen results in Findings 1–4 |
| `step43d5_prompt_compare.py` | InternVL3-8B run of the same `v_dynamic_cot_viewer_perspective` variant tested on Qwen, to check whether a fix motivated by a Qwen-specific issue (left/right self-contradiction) transfers to InternVL3. It doesn't — see Finding 5. | Used for the InternVL3 viewer-perspective regression result |

**If you only need one InternVL3 script and one Qwen script going forward**, use
`step43d3_prompt_compare.py` (or its later off-screen-improved sibling) and
`step43e2_prompt_compare_qwen.py` — everything before those has been superseded by bug fixes
or prompt revisions.

---

## What the code actually does (common structure across all versions)

1. **Looks up a hand-picked frame** by `(clip_id, frame_number, subject_id)` directly from the
   VAT annotation `.txt` files and corresponding image folder — no manifest/CSV dependency,
   so it works for clips outside the production `clip_selected.csv` (e.g. Project Runway,
   BTS/Jimmy Fallon).
2. **Builds 1–3 prompt variants** for that frame, each asking the VLM to output a normalised
   gaze-target coordinate (`GAZE_XY: (x, y)`) or an off-screen sentinel.
3. **Runs inference** (InternVL3-8B or Qwen2.5-VL-7B-Instruct, depending on the script) and
   parses the response.
4. **Prints ground truth alongside every variant's prediction and raw response text**, so
   results can be eyeballed and compared frame-by-frame in the console — there is no
   automated scoring; everything in the findings doc was checked by hand against this printed
   output.

## Known limitations of this code (intentional, given its throwaway/diagnostic purpose)

- No error handling beyond a basic `SKIP` if a frame/clip can't be resolved — this is fine for
  a fixed, manually-curated `TARGET_FRAMES` list, but not robust enough for unattended/batch use.
- `BASE_PATH` is hardcoded to a local Windows path.
- No results are saved to disk — output is console-only (hence this week's re-run when results
  weren't copied out in time).
- Frame numbers in `TARGET_FRAMES` were manually transcribed while reviewing footage by eye,
  and have a history of being wrong (see `step43d2` → `step43d3` fix) — if extending
  `TARGET_FRAMES` with new clips, verify frame numbers exist in the actual annotation file
  before trusting any output for them.
