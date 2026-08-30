"""
dedupe_ucolaeo_cross_split.py

Fixes UCO-LAEO's cross-split H5 duplication (confirmed: ~2,858 (show,
fname, subject) keys appear in more than one of the three source H5
files -- e.g. video got05 sitting in both train and test). Confirmed
this predates every script in this project (it's a property of the
released H5 files themselves), and that predictions for the resulting
duplicate rows agree exactly (0 disagreements, checked directly) -- so
this is safe deduplication of redundant-but-consistent data, not a
silent discard of genuinely conflicting information.

RULE, MADE EXPLICIT: when a key spans multiple splits, the SURVIVING
split is whichever is highest priority in test > val > train order.
Not arbitrary -- this exactly matches the behaviour
ucolaeo_temporal_utils.py's build_person_timelines() already had, by
accident, from its train->val->test loop order (last write wins).
Made explicit and applied consistently everywhere here, rather than
leaving one file correct by coincidence and every other file
inconsistent with it.

RUN IN THIS EXACT ORDER (all handled automatically in one execution --
see __main__ below, no need to run this script more than once) -- steps
2/2b/3 depend on step 1's corrected split assignment, not the original
(duplicated) manifest's own values:
  1. Dedupe frame_manifest_ucolaeo.csv -> establishes the canonical
     (show, fname, subject) -> split mapping.
  2. Dedupe labels_ucolaeo_full.jsonl -> one row per key. This file has
     no "split" field of its own (confirmed against a real row earlier
     this project) and content is already confirmed identical across
     duplicates, so keeping the first occurrence is safe.
  2b. Dedupe labels_ucolaeo_hybrid_targeted.jsonl -- added after
     confirming this file is structurally identical to labels_full for
     UCO-LAEO specifically (100% social, hybrid routing never diverges
     from InternVL3-alone -- confirmed via step46's own pipeline log),
     so it almost certainly inherits the exact same duplication. Same
     keep-first treatment as step 2.
  3. Dedupe mtgs_teacher_ucolaeo.jsonl -> one row per key, AND each
     surviving row's own "split" field is RE-STAMPED from step 1's
     corrected manifest. The original export ran against the
     un-deduped manifest, so a surviving row's split value may not
     match the new canonical assignment otherwise -- this is the one
     step that's NOT just "drop duplicates," so don't skip it.
"""

import csv
import json

MANIFEST_IN = "frame_manifest_ucolaeo.csv"
MANIFEST_OUT = "frame_manifest_ucolaeo_deduped.csv"
LABELS_IN = "labels_ucolaeo_full.jsonl"
LABELS_OUT = "labels_ucolaeo_full_deduped.jsonl"
HYBRID_LABELS_IN = "labels_ucolaeo_hybrid_targeted.jsonl"
HYBRID_LABELS_OUT = "labels_ucolaeo_hybrid_targeted_deduped.jsonl"
MTGS_IN = "mtgs_teacher_ucolaeo.jsonl"
MTGS_OUT = "mtgs_teacher_ucolaeo_deduped.jsonl"

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = higher priority


def dedupe_manifest():
    rows_by_key = {}
    fieldnames = None
    n_total = 0
    with open(MANIFEST_IN, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            n_total += 1
            key = (row["show"], row["fname"], row["subject"])
            if key not in rows_by_key:
                rows_by_key[key] = row
            else:
                existing = rows_by_key[key]
                if SPLIT_PRIORITY[row["split"]] < SPLIT_PRIORITY[existing["split"]]:
                    rows_by_key[key] = row

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_key.values())

    split_map = {k: v["split"] for k, v in rows_by_key.items()}
    print(f"[manifest] {n_total} rows -> {len(rows_by_key)} unique keys "
          f"({n_total - len(rows_by_key)} duplicates resolved via "
          f"test>val>train priority). Written to {MANIFEST_OUT}.")
    return split_map


def dedupe_simple(path_in, path_out, label):
    """For files with no split field of their own, or where content is
    already confirmed identical across duplicates -- keeps the first
    occurrence per key."""
    seen = set()
    n_total = 0
    n_kept = 0
    with open(path_in, encoding="utf-8") as f_in, \
         open(path_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            n_total += 1
            r = json.loads(line)
            key = (r["show"], r["fname"], r["subject"])
            if key in seen:
                continue
            seen.add(key)
            f_out.write(line if line.endswith("\n") else line + "\n")
            n_kept += 1
    print(f"[{label}] {n_total} rows -> {n_kept} unique keys "
          f"({n_total - n_kept} duplicates dropped). Written to {path_out}.")


def dedupe_mtgs_with_split_restamp(split_map):
    """Same as dedupe_simple, but also re-stamps each surviving row's
    own "split" field from the CORRECTED manifest -- the original
    export ran against the un-deduped manifest, so a surviving row's
    split may not match the new canonical assignment otherwise."""
    seen = set()
    n_total = 0
    n_kept = 0
    n_restamped = 0
    n_no_manifest_match = 0
    with open(MTGS_IN, encoding="utf-8") as f_in, \
         open(MTGS_OUT, "w", encoding="utf-8") as f_out:
        for line in f_in:
            n_total += 1
            r = json.loads(line)
            key = (r["show"], r["fname"], r["subject"])
            if key in seen:
                continue
            seen.add(key)

            correct_split = split_map.get(key)
            if correct_split is None:
                n_no_manifest_match += 1
            elif r.get("split") != correct_split:
                r["split"] = correct_split
                n_restamped += 1

            f_out.write(json.dumps(r) + "\n")
            n_kept += 1

    print(f"[mtgs_teacher] {n_total} rows -> {n_kept} unique keys "
          f"({n_total - n_kept} duplicates dropped, {n_restamped} "
          f"split fields corrected, {n_no_manifest_match} keys with no "
          f"manifest match -- investigate directly if this is nonzero). "
          f"Written to {MTGS_OUT}.")


if __name__ == "__main__":
    print("Step 1: deduping manifest, establishing canonical split map...")
    split_map = dedupe_manifest()

    print("\nStep 2: deduping labels_ucolaeo_full.jsonl (content already "
          "confirmed identical across duplicates -- keep-first is safe)...")
    dedupe_simple(LABELS_IN, LABELS_OUT, "labels_full")

    print("\nStep 2b: deduping labels_ucolaeo_hybrid_targeted.jsonl -- "
          "structurally identical to labels_full for UCO-LAEO specifically "
          "(100% social, hybrid routing never diverges from InternVL3-alone, "
          "confirmed via step46's own pipeline log), so almost certainly "
          "inherits the exact same duplication. Fixed here rather than left "
          "as a mismatch against the now-clean labels_full...")
    dedupe_simple(HYBRID_LABELS_IN, HYBRID_LABELS_OUT, "labels_hybrid_targeted")

    print("\nStep 3: deduping mtgs_teacher_ucolaeo.jsonl, re-stamping split...")
    dedupe_mtgs_with_split_restamp(split_map)

    print("\nDone. Recommended before trusting these: rerun the same "
          "duplicate-count checks against each *_deduped file (should "
          "report 0 duplicates everywhere), and spot-check a few "
          "restamped mtgs_teacher rows by hand against the new manifest.")
