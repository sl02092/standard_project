#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_teacher_noleak.py — produce no-leak / always-DINO variants of the
step40 and step46 teacher pipelines, for the two §7.4 experiments.

WHAT IT DOES
────────────
Reads each step40_*.py / step46_*.py you point it at and writes a NEW file
alongside it with "_noleak" (step40) or "_alwaysdino" (step46) before the
.py. ORIGINALS ARE NEVER MODIFIED. It refuses to write anything unless every
anchor it expects is found exactly once, so a file that has drifted from the
frozen versions fails loudly instead of being half-patched.

step40 (InternVL3-alone)  — the prompt-leakage fix
    1. build_social_prompt() no longer supplies PERSON B's head box or face
       centre, and no longer instructs the model to answer with that centre.
       §3.2.3 measured what the old prompt produced: the supplied coordinate
       returned unchanged on 88.1% of social frames and ~84.9% of object
       frames, where it is wrong. This is §7.4 item 2's specified experiment.
    2. TEACHER_VERSION bumped, so a label can never be mistaken for one from
       the frozen production run.
    3. Every output filename gains a tag, so the run cannot resume from — or
       overwrite — the frozen labels.

step46 (InternVL3 + Grounding DINO) — the always-take-the-detection variant
    1. The SAME prompt fix, applied to its social path. Without this the two
       conditions stop sharing social labels (98.3% shared today, §5.2) and
       the "they differ only on object frames" comparison is destroyed.
    2. BOX_THRESHOLD / TEXT_THRESHOLD driven to 0, so Grounding DINO's best
       box is always accepted and the no-box GT fallback effectively never
       fires.
    3. The remaining GT fallback on object frames — InternVL3 claiming
       off-screen — records the teacher's actual (-1,-1) claim instead of
       substituting ground truth, so the condition is a SINGLE generating
       process. That uniformity is the entire point of §7.4 item 1: a
       condition that is uniform but noisier. Any residual GT puts the
       mixture back and costs you the experiment.
    4. TEACHER_VERSION_SOCIAL / _HYBRID bumped and filenames tagged, as above.

USAGE
─────
    python patch_teacher_noleak.py step40_*.py step46_*.py
    python patch_teacher_noleak.py --outdir patched/ *.py

    --tag NAME   filename tag (default: noleak)
    --outdir DIR write patched files here instead of beside the originals
    --check      report what would change; write nothing

RUNTIME SWITCHES in the patched files (both default to the new behaviour):
    TEACHER_RERUN_TAG    filename tag, overrides the baked-in one
    HYBRID_ALWAYS_DINO   "0" restores thresholds and GT fallback (step46)
"""
import argparse
import os
import re
import sys

NEW_VERSION_SOCIAL = "internvl3-8b-v5-noleak"
NEW_VERSION_HYBRID = "internvl3+gdino-base-v3-alwaysdino"

# ── anchor 1: the leaking prompt body ────────────────────────────────
# Matches both spellings (step40's multi-line docstring and step46's
# one-liner) by anchoring on the code between the signature and the end of
# the returned f-string, which is byte-identical in every file.
OLD_PROMPT_RE = re.compile(
    r"""    ax1, ay1, ax2, ay2 = primary_box\n"""
    r"""(?:\s*\n)?    bx1, by1, bx2, by2 = secondary_box\n"""
    r""".*?"""
    r"""After the coordinate, add one sentence explaining why\."{3}""",
    re.DOTALL,
)

NEW_PROMPT = '''    ax1, ay1, ax2, ay2 = primary_box

    # NO-LEAK v5 (2026-09-18). The frozen v4 prompt supplied PERSON B's head
    # box AND face centre and told the model to answer with that centre when
    # A looked at B. Section 3.2.3 measured the consequence: the supplied
    # coordinate came back unchanged on 88.1% of social frames and ~84.9% of
    # object frames, where it is wrong -- so the condition was measuring the
    # model's willingness to copy, not its gaze competence.
    #
    # This version withholds the coordinate AND the box it could be trivially
    # derived from, which is Section 7.4 item 2's specified experiment. The
    # LABELLING CONVENTION is deliberately kept -- a social target is still
    # the other person's face centre -- so the output stays comparable with
    # ground truth; only the answer is withheld.
    #
    # secondary_box is still accepted so that every call site is unchanged,
    # but it is deliberately unused. Do not reintroduce it.
    na = (round(ax1/img_w,3), round(ay1/img_h,3),
          round(ax2/img_w,3), round(ay2/img_h,3))

    return f"""Gaze estimation task. Image: {img_w}x{img_h}px.

PERSON A (predict gaze): head box (normalised) ({na[0]},{na[1]}) to ({na[2]},{na[3]})

YOUR FIRST LINE MUST BE:
GAZE_XY: (x, y)

Rules:
- Work out where Person A is looking from their head pose and eye direction, then locate that point in the image yourself.
- If A is looking at another person, answer with the centre of THAT person's face.
- If A is looking at an object, answer with the centre of that object.
- If A is looking outside the frame, answer (-1,-1).
- x,y are normalised 0.0-1.0 (0,0=top-left, 1,1=bottom-right)

After the coordinate, add one sentence explaining why."""'''

# ── anchor 2: output filenames (last `    LOG_FILE = ...` of the if/else) ──
LOGFILE_RE = re.compile(r"^    LOG_FILE\s*=.*$", re.MULTILINE)

FILENAME_TAG_BLOCK = '''

# ── RERUN GUARD (patch_teacher_noleak.py) ────────────────────────────
# Every output filename is tagged so this run can NEITHER resume from NOR
# overwrite the frozen production labels. Without this, recover_progress_
# from_jsonl() reads the OLD labels file, concludes every pair is already
# done, and the job exits having written nothing -- silently.
_RERUN_TAG    = os.environ.get("TEACHER_RERUN_TAG", "{tag}")
LABELS_FILE   = LABELS_FILE.replace(".jsonl", f"_{{_RERUN_TAG}}.jsonl")
PROGRESS_FILE = PROGRESS_FILE.replace(".json",  f"_{{_RERUN_TAG}}.json")
LOG_FILE      = LOG_FILE.replace(".log",        f"_{{_RERUN_TAG}}.log")
'''

# ── anchor 3: step46 thresholds ──────────────────────────────────────
OLD_THRESHOLDS = "BOX_THRESHOLD   = 0.30\nTEXT_THRESHOLD  = 0.25"

NEW_THRESHOLDS = '''# ALWAYS-DINO (patch_teacher_noleak.py). Driving both thresholds to 0 makes
# post_process_grounded_object_detection return its full candidate set, so
# the argmax below always yields a box and the "no box found" GT fallback
# effectively never fires. That is what makes this condition a SINGLE
# generating process -- Section 7.4 item 1's "uniform, but noisier".
# Set HYBRID_ALWAYS_DINO=0 to restore the frozen 0.30/0.25 behaviour.
ALWAYS_DINO     = os.environ.get("HYBRID_ALWAYS_DINO", "1") == "1"
BOX_THRESHOLD   = 0.0 if ALWAYS_DINO else 0.30
TEXT_THRESHOLD  = 0.0 if ALWAYS_DINO else 0.25'''

# ── anchor 4: step46 object-frame label assignment ───────────────────
OLD_OBJ_BLOCK = """        if pred_x is not None:
            label_source = "teacher_hybrid"
            confidence   = "PARSED"
            final_x, final_y = pred_x, pred_y
            n_obj_ok += 1
        else:
            label_source = "gt_fallback"
            confidence   = "GT_FALLBACK"
            final_x, final_y = gt_x_norm, gt_y_norm
            n_obj_fail += 1"""

NEW_OBJ_BLOCK = '''        if pred_x is not None:
            label_source = "teacher_hybrid"
            confidence   = "PARSED"
            final_x, final_y = pred_x, pred_y
            n_obj_ok += 1
        elif ALWAYS_DINO and object_phrase is None:
            # InternVL3 claimed off-screen on a frame ground truth calls
            # object-directed. The frozen pipeline substituted GT here.
            # Doing that reintroduces a second generating process into the
            # one condition whose whole purpose is to have only one, so
            # record the teacher's actual claim instead -- the same
            # reasoning as step40 v5's off-screen fix. Counted separately
            # and reported at the end; if this count is large the
            # uniformity claim is weaker than it looks.
            label_source = "teacher_offscreen"
            confidence   = "PARSED"
            final_x, final_y = -1.0, -1.0
            n_obj_offscreen_kept += 1
        else:
            label_source = "gt_fallback"
            confidence   = "GT_FALLBACK"
            final_x, final_y = gt_x_norm, gt_y_norm
            n_obj_fail += 1'''

# counter init + summary line, so n_obj_offscreen_kept exists and is reported
OLD_COUNTER = "    n_internvl_offscreen, n_no_box = 0, 0"
NEW_COUNTER = ("    n_internvl_offscreen, n_no_box = 0, 0\n"
               "    n_obj_offscreen_kept = 0   # ALWAYS-DINO: teacher off-screen claims kept, not GT-substituted")

# ── anchor 5: step46 RESUME BUG (fires only on a resumed run) ────────
# step46 rebinds `df` to the not-yet-done rows and then builds the frame
# lookup from that filtered frame. step40 keeps the full frame and iterates
# `remaining`, which is correct. The consequence in step46: on a resume, a
# frame whose OTHER subject was completed before the kill now looks solo,
# takes the gt_fallback_solo branch, and writes GROUND TRUTH as the label
# for a teacher row. Silent, and exactly the leakage class the v2/v5 fixes
# were about. Never fires on a clean single-shot run, which is why the VAT
# production run is unaffected.
OLD_RESUME = ('    df = df[~df["pair_id"].isin(completed)].reset_index(drop=True)\n'
              '    log(f"Remaining: {len(df)} pairs")')
NEW_RESUME = ('    # RESUME FIX (patch_teacher_noleak.py): keep the FULL manifest for the\n'
              '    # frame lookup. Filtering it before build_frame_lookup() makes any frame\n'
              '    # whose co-subject finished before the kill look solo on resume, which\n'
              '    # sends a teacher row down the gt_fallback_solo path and writes ground\n'
              '    # truth as its label. step40 already does it this way.\n'
              '    df_all = df\n'
              '    df = df[~df["pair_id"].isin(completed)].reset_index(drop=True)\n'
              '    log(f"Remaining: {len(df)} pairs")')

OLD_LOOKUP = "    frame_lookup = build_frame_lookup(df)  # v3: pre-built lookup, see above"
NEW_LOOKUP = ("    frame_lookup = build_frame_lookup(df_all)  # RESUME FIX: full manifest,\n"
              "    # not the post-resume remainder -- see the note at the resume filter.")

OLD_SUMMARY = '    log(f"    of which GDINO found no box:        {n_no_box}")'
NEW_SUMMARY = ('    log(f"    of which GDINO found no box:        {n_no_box}")\n'
               '    log(f"  Object teacher off-screen kept (NOT GT): {n_obj_offscreen_kept}")\n'
               '    log(f"  ALWAYS_DINO={ALWAYS_DINO}  BOX_THRESHOLD={BOX_THRESHOLD}  '
               'TEXT_THRESHOLD={TEXT_THRESHOLD}")\n'
               '    if n_obj_fail:\n'
               '        log(f"  WARNING: {n_obj_fail} object rows still took GT -- this condition is '
               'not a single generating process. Check them before claiming uniformity.")')


OLD_VERSION_SOCIAL = "internvl3-8b-v4-prompt"
OLD_VERSION_HYBRID = "internvl3+gdino-base-v2-targeted"


class PatchError(RuntimeError):
    pass


def _set_version(text, name, old_value, new_value):
    """Retag a TEACHER_VERSION* constant.

    Anchored on the ASSIGNMENT, not the bare string literal: step46's module
    docstring quotes both tags in its OUTPUT section, so the literal occurs
    twice and a naive count-must-be-1 check fails on every step46 file. The
    assignment occurs exactly once. Any remaining occurrences are docstring
    or comment references and are swept afterwards so the patched file's own
    documentation does not describe the frozen run's tags.
    """
    pat = re.compile(rf'^({re.escape(name)}\s*=\s*)"[^"]*"', re.MULTILINE)
    hits = list(pat.finditer(text))
    if len(hits) != 1:
        raise PatchError(f"{name}: expected 1 assignment, found {len(hits)}")
    h = hits[0]
    text = text[:h.start()] + h.group(1) + f'"{new_value}"' + text[h.end():]
    n_doc = text.count(f'"{old_value}"')
    if n_doc:
        text = text.replace(f'"{old_value}"', f'"{new_value}"')
    return text, n_doc


def _sub_once(text, pattern, repl, what):
    """Replace exactly once. Anything else is a hard failure."""
    if isinstance(pattern, str):
        n = text.count(pattern)
        if n != 1:
            raise PatchError(f"{what}: expected 1 occurrence, found {n}")
        return text.replace(pattern, repl)
    hits = list(pattern.finditer(text))
    if len(hits) != 1:
        raise PatchError(f"{what}: expected 1 match, found {len(hits)}")
    return text[:hits[0].start()] + repl + text[hits[0].end():]


def patch_step40(src, tag):
    changes = []
    out = _sub_once(src, OLD_PROMPT_RE, NEW_PROMPT, "leaking prompt body")
    changes.append("prompt: PERSON B box + face centre withheld")

    out, n_doc = _set_version(out, "TEACHER_VERSION",
                              OLD_VERSION_SOCIAL, NEW_VERSION_SOCIAL)
    changes.append(f"TEACHER_VERSION -> {NEW_VERSION_SOCIAL}"
                   + (f" (+{n_doc} doc reference(s))" if n_doc else ""))

    last = list(LOGFILE_RE.finditer(out))
    if not last:
        raise PatchError("output filenames: no `    LOG_FILE =` line found")
    at = last[-1].end()
    out = out[:at] + FILENAME_TAG_BLOCK.format(tag=tag) + out[at:]
    changes.append(f"output filenames tagged _{tag}")
    return out, changes


def patch_step46(src, tag):
    changes = []
    out = _sub_once(src, OLD_PROMPT_RE, NEW_PROMPT, "leaking prompt body (social path)")
    changes.append("social prompt: PERSON B box + face centre withheld")

    out, n_soc = _set_version(out, "TEACHER_VERSION_SOCIAL",
                              OLD_VERSION_SOCIAL, NEW_VERSION_SOCIAL)
    out, n_hyb = _set_version(out, "TEACHER_VERSION_HYBRID",
                              OLD_VERSION_HYBRID, NEW_VERSION_HYBRID)
    changes.append(f"versions -> {NEW_VERSION_SOCIAL} / {NEW_VERSION_HYBRID}"
                   + (f" (+{n_soc + n_hyb} doc reference(s))" if (n_soc + n_hyb) else ""))

    out = _sub_once(out, OLD_THRESHOLDS, NEW_THRESHOLDS, "BOX/TEXT thresholds")
    changes.append("BOX_THRESHOLD/TEXT_THRESHOLD -> 0.0 (env-switchable)")

    out = _sub_once(out, OLD_COUNTER, NEW_COUNTER, "off-screen counter init")
    out = _sub_once(out, OLD_OBJ_BLOCK, NEW_OBJ_BLOCK, "object-frame label assignment")
    out = _sub_once(out, OLD_SUMMARY, NEW_SUMMARY, "summary log")
    changes.append("object GT fallback -> teacher's own (-1,-1) claim; counted + warned")

    out = _sub_once(out, OLD_RESUME, NEW_RESUME, "resume filter (frame-lookup scope)")
    out = _sub_once(out, OLD_LOOKUP, NEW_LOOKUP, "frame lookup build")
    changes.append("RESUME BUG fixed: frame lookup built from the full manifest")

    last = list(LOGFILE_RE.finditer(out))
    if not last:
        raise PatchError("output filenames: no `    LOG_FILE =` line found")
    at = last[-1].end()
    out = out[:at] + FILENAME_TAG_BLOCK.format(tag=tag) + out[at:]
    changes.append(f"output filenames tagged _{tag}")
    return out, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tag", default="noleak")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    ok = fail = 0
    for path in a.files:
        base = os.path.basename(path)
        if base.startswith("step40"):
            kind, suffix = "step40", "_noleak"
        elif base.startswith("step46"):
            kind, suffix = "step46", "_alwaysdino"
        else:
            print(f"SKIP  {base}  (not a step40_/step46_ file)")
            continue

        src = open(path, encoding="utf-8").read()
        try:
            out, changes = (patch_step40 if kind == "step40" else patch_step46)(src, a.tag)
            compile(out, base, "exec")          # syntax-check before writing
        except PatchError as e:
            print(f"FAIL  {base}\n        {e}")
            fail += 1
            continue
        except SyntaxError as e:
            print(f"FAIL  {base}\n        patched file does not compile: {e}")
            fail += 1
            continue

        dest_dir = a.outdir or os.path.dirname(os.path.abspath(path))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, base[:-3] + suffix + ".py")
        print(f"OK    {base}  ->  {os.path.basename(dest)}")
        for c in changes:
            print(f"        - {c}")
        if not a.check:
            with open(dest, "w", encoding="utf-8") as f:
                f.write(out)
        ok += 1

    print(f"\n{ok} patched, {fail} failed"
          + ("   (--check: nothing written)" if a.check else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
