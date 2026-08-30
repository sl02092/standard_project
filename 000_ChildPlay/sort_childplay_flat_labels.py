"""
sort_childplay_flat_labels.py — Sorts an already-built labels_childplay_gt.jsonl
by (clip, frame_idx, subject), matching VAT's own on-disk convention
(step40's df.sort_values(["show","clip","subject","fname"]) before writing).

WHY: the original extraction script preserves manifest first-encounter order
(a dict's insertion order), not an explicit sort -- functionally harmless for
training (step52-family shuffles at train time regardless of on-disk order),
but makes manual inspection hard (same-clip, same-frame rows aren't adjacent)
and is a latent risk for any future code that assumes sorted input without
checking. Cheap to fix as a single re-sort pass over the existing output --
no need to re-run the manifest scan or the CSV joins.

USAGE:
    python sort_childplay_flat_labels.py labels_childplay_gt.jsonl \
        --out labels_childplay_gt_sorted.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default=None,
                     help="default: <infile> with _sorted before .jsonl")
    args = ap.parse_args()
    out_path = args.out or args.infile.replace(".jsonl", "_sorted.jsonl")

    with open(args.infile, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    rows.sort(key=lambda r: (r["clip"], r["frame_idx"], r["subject"]))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Sorted {len(rows)} rows -> {out_path}")
    print("Same-frame, multi-subject rows are now adjacent -- "
          "spot-checking a multi-person frame should be easy to find now: "
          "look for consecutive rows sharing the same clip AND frame_idx.")


if __name__ == "__main__":
    main()
