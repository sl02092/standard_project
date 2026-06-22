# Quick sanity check — run this before step5
import json, os

LABELS_FILE = "labels_test.jsonl" if os.path.exists("labels_test.jsonl") and \
              not os.path.exists("labels_full.jsonl") else "labels_full.jsonl"

records = [json.loads(l) for l in open(LABELS_FILE) if l.strip()]
print(f"Checking      : {LABELS_FILE}")
#records = [json.loads(l) for l in open("labels_full.jsonl") if l.strip()]

sources     = {}
confidence  = {}
versions    = {}
for r in records:
    sources[r["label_source"]]              = sources.get(r["label_source"], 0) + 1
    confidence[r["confidence"]]             = confidence.get(r["confidence"], 0) + 1
    versions[str(r["teacher_version"])]     = versions.get(str(r["teacher_version"]), 0) + 1

print(f"Total labels     : {len(records)}")
print(f"Sources          : {sources}")
print(f"Confidence       : {confidence}")
print(f"Teacher versions : {versions}")
print(f"Any None pred_x  : {sum(1 for r in records if r['pred_x'] is None)}")
print(f"Any None frame_idx: {sum(1 for r in records if r['frame_idx'] is None)}")