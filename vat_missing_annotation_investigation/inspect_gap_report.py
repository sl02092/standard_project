"""
inspect_gap_report.py — quick filter over vat_temporal_gap_report.csv
to pull the exact missing frame filenames for a given show, so you can
go straight to the annotation .txt files and check the frame numbering.

Usage:
    python inspect_gap_report.py vat_temporal_gap_report.csv "Sound of Music"
"""
import sys
import csv
from collections import defaultdict

def main():
    if len(sys.argv) < 3:
        print("Usage: python inspect_gap_report.py <gap_report.csv> <show name>")
        return

    report_path, show_filter = sys.argv[1], sys.argv[2]

    by_clip_subject = defaultdict(set)
    with open(report_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["show"] != show_filter:
                continue
            key = (row["clip"], row["subject"])
            for fname in row["missing_fnames"].split("|"):
                by_clip_subject[key].add(fname)

    if not by_clip_subject:
        print(f"No gap rows found for show == '{show_filter}'.")
        return

    for (clip, subject), fnames in sorted(by_clip_subject.items()):
        sorted_fnames = sorted(fnames)
        print(f"\nClip: {clip}  |  Subject: {subject}")
        print(f"  {len(sorted_fnames)} distinct missing frame(s):")
        print(f"  {sorted_fnames}")
        print(f"  -> Open {subject}.txt for this clip and check whether these")
        print(f"     frame numbers are simply absent as rows (annotation gap),")
        print(f"     or whether the file jumps over them in its frame numbering.")

if __name__ == "__main__":
    main()
