"""
verify_vat_temporal_completeness.py — checks vat_temporal_manifest.jsonl
against the concrete reference figures recorded in
VAT_temporal_findings_README.md (214 qualifying clips, 21,397 candidate
windows, 64,191 total rows, 59,138 valid rows) -- resolves the "may/may
not have the entire dataset" open question with real numbers rather than
continued uncertainty.

Also closes a second, separate open item from that same README while
we're here: "Per-horizon invalid-rate breakdown -- flagged as worth
checking but not yet actually pulled from the manifest." Same data,
already being read for the completeness check -- no reason not to
answer both at once.

Usage:
    python3 verify_vat_temporal_completeness.py [manifest_path]
    (defaults to vat_temporal_manifest.jsonl in the current directory)
"""

import sys
import json
from collections import Counter, defaultdict

REFERENCE = {
    "qualifying_clips": 214,
    "candidate_windows": 21397,   # distinct (clip, subject) windows, before x3 horizons
    "total_rows": 64191,          # candidate_windows x 3 horizons
    "valid_rows": 59138,
}


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "vat_temporal_manifest.jsonl"

    clip_ids = set()
    windows_seen = set()  # (clip_id, subject_id, context_frame_paths) -- the LAST
                          # element disambiguates overlapping stride-4 windows for
                          # the same subject in the same clip; a shared context
                          # tuple across different horizon rows correctly collapses
                          # to one window (found and fixed 2026-08-08 -- the original
                          # (clip_id, subject_id)-only key undercounted badly, since
                          # one subject can anchor many windows per clip)
    total_rows = 0
    valid_rows = 0
    rows_by_horizon = Counter()
    valid_by_horizon = Counter()
    invalid_by_horizon = Counter()

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1

            clip_ids.add(row["clip_id"])
            windows_seen.add((row["clip_id"], row["subject_id"], tuple(row["context_frame_paths"])))

            hf = row.get("horizon_frames")
            rows_by_horizon[hf] += 1
            if row["window_valid"]:
                valid_rows += 1
                valid_by_horizon[hf] += 1
            else:
                invalid_by_horizon[hf] += 1

    print("=" * 70)
    print("VAT TEMPORAL MANIFEST — COMPLETENESS CHECK")
    print("=" * 70)
    print(f"Manifest: {manifest_path}\n")

    def compare(label, actual, expected):
        status = "MATCH" if actual == expected else "MISMATCH"
        print(f"  {label:<28} actual={actual:<10} expected={expected:<10} [{status}]")

    compare("Distinct clips", len(clip_ids), REFERENCE["qualifying_clips"])
    compare("Distinct (clip,subject) windows", len(windows_seen), REFERENCE["candidate_windows"])
    compare("Total rows", total_rows, REFERENCE["total_rows"])
    compare("Valid rows", valid_rows, REFERENCE["valid_rows"])

    all_match = (
        len(clip_ids) == REFERENCE["qualifying_clips"]
        and len(windows_seen) == REFERENCE["candidate_windows"]
        and total_rows == REFERENCE["total_rows"]
        and valid_rows == REFERENCE["valid_rows"]
    )
    print()
    if all_match:
        print("ALL FOUR figures match the README's reference exactly -- this manifest "
              "is confirmed complete, not a partial/interrupted build.")
    else:
        print("At least one figure does NOT match -- investigate before trusting this "
              "manifest for real results. A distinct-clips mismatch alone (with rows "
              "roughly proportional) would suggest a genuinely smaller/larger clip "
              "population; a rows mismatch with clips matching would suggest a "
              "different issue (e.g. a horizon or filtering change since the README "
              "was written).")

    print("\n" + "=" * 70)
    print("PER-HORIZON INVALID-RATE BREAKDOWN")
    print("(closes a second, separate open item from the README -- expectation")
    print(" stated there: 500ms/12-frame should show a HIGHER invalid rate than")
    print(" 100ms/2-frame, since less clip remains after a longer context+horizon)")
    print("=" * 70)
    label_map = {2: "100ms (2 frames)", 7: "300ms (7 frames)", 12: "500ms (12 frames)"}
    for hf in sorted(rows_by_horizon):
        total_h = rows_by_horizon[hf]
        valid_h = valid_by_horizon[hf]
        invalid_h = invalid_by_horizon[hf]
        rate = 100 * invalid_h / total_h if total_h else 0
        label = label_map.get(hf, f"horizon_frames={hf}")
        print(f"  {label:<20} total={total_h:<8} valid={valid_h:<8} "
              f"invalid={invalid_h:<8} invalid_rate={rate:.2f}%")

    rates = {hf: (100 * invalid_by_horizon[hf] / rows_by_horizon[hf] if rows_by_horizon[hf] else 0)
             for hf in rows_by_horizon}
    if 2 in rates and 12 in rates:
        if rates[12] > rates[2]:
            print(f"\n  CONFIRMED: 500ms invalid rate ({rates[12]:.2f}%) > 100ms invalid "
                  f"rate ({rates[2]:.2f}%) -- matches the README's own stated expectation.")
        else:
            print(f"\n  UNEXPECTED: 500ms invalid rate ({rates[12]:.2f}%) is NOT higher than "
                  f"100ms's ({rates[2]:.2f}%) -- worth understanding why before assuming "
                  f"the horizon logic is behaving as expected.")


if __name__ == "__main__":
    main()
