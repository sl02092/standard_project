import json, csv
from collections import Counter

manifest_keys = {}
with open('frame_manifest_vacation.csv') as f:
    for row in csv.DictReader(f):
        manifest_keys[(row['show'], row['fname'], row['subject'])] = row['gaze_type']

written = set()
with open('labels_vacation_full.jsonl') as f:
    for line in f:
        r = json.loads(line)
        written.add((r['show'], r['fname'], r['subject']))

missing = set(manifest_keys) - written
print(f"Total missing: {len(missing)}")
print("Missing by gaze_type:", Counter(manifest_keys[k] for k in missing))
print("Full population by gaze_type:", Counter(manifest_keys.values()))