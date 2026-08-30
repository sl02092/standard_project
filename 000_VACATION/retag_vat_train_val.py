#!/usr/bin/env python3
"""Splits VAT's merged mtgs_population=="main" rows into "train"/"val"
using MTGS's own VAL_SHOWS carve-out (3 shows, confirmed via direct code
review of videoattentiontarget_temporal.py: Orange_is_the_New_Black,
Before_Sunrise, Project_Runway -- drawn from within VAT's official train
split). Renames "testshows" -> "test" so VAT ends up using the exact
same train/val/test vocabulary as ChildPlay and VACATION.

Handles both space and underscore show-name conventions defensively
(this project has used both at different points -- "show_us" vs
"show_spaces") and prints real sample values before trusting anything,
same discipline as every naming-convention check today.

Usage:
    python3 retag_vat_train_val.py frame_manifest_vat.csv
    python3 retag_vat_train_val.py labels_vat_full.jsonl
    python3 retag_vat_train_val.py labels_vat_hybrid_targeted.jsonl
    python3 retag_vat_train_val.py mtgs_teacher_vat_merged.jsonl
"""
import sys
import csv
import json

VAL_SHOWS_RAW = ["Orange_is_the_New_Black", "Before_Sunrise", "Project_Runway"]


def normalize(show):
    return show.strip().lower().replace(" ", "_") if show else show


VAL_SHOWS_NORM = {normalize(s) for s in VAL_SHOWS_RAW}


def retag(row):
    if row.get("mtgs_population") == "testshows":
        row["mtgs_population"] = "test"
    elif row.get("mtgs_population") == "main":
        row["mtgs_population"] = "val" if normalize(row.get("show")) in VAL_SHOWS_NORM else "train"
    return row


def main():
    path = sys.argv[1]
    is_csv = path.lower().endswith(".csv")

    if is_csv:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    print(f"Total rows: {len(rows)}")

    # Verification before trusting anything -- show real 'main'-population
    # show values and how they resolve.
    main_shows = sorted({r["show"] for r in rows if r.get("mtgs_population") == "main"})
    matched_val_shows = [s for s in main_shows if normalize(s) in VAL_SHOWS_NORM]
    print(f"Shows in 'main' population matching VAL_SHOWS: {matched_val_shows}")
    if len(matched_val_shows) != 3:
        print(f"WARNING: expected exactly 3 matching shows, found "
              f"{len(matched_val_shows)}. Check show-name formatting "
              f"before trusting the output below -- printing all "
              f"'main' show values containing a likely fragment for "
              f"manual comparison:")
        for s in main_shows:
            if any(frag in s.lower() for frag in ("orange", "sunrise", "runway")):
                print(f"  candidate: {s!r}")

    for r in rows:
        retag(r)

    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        counts[r["mtgs_population"]] = counts.get(r["mtgs_population"], 0) + 1
    print(f"\nFinal counts: {counts}")

    out_path = path.rsplit(".", 1)[0] + "_RETAGGED." + path.rsplit(".", 1)[1]
    if is_csv:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
