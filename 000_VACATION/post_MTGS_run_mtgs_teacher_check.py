import json, csv
from collections import Counter

# Check 1: does the duplication already exist in the STATIC MANIFEST,
# i.e. does it predate step40 entirely?
keys = []
with open('frame_manifest_vacation.csv') as f:
    for row in csv.DictReader(f):
        keys.append((row['show'], row['fname'], row['subject']))
c = Counter(keys)
print(f"Manifest: {len(keys)} rows, {len(c)} unique keys, {len(keys)-len(c)} duplicates")

# Check 2: did MTGS's own OUTPUT inherit the duplication, as expected?
keys2 = []
with open('mtgs_teacher_vacation.jsonl') as f:
    for line in f:
        r = json.loads(line)
        keys2.append((r['show'], r['fname'], r['subject']))
c2 = Counter(keys2)
print(f"MTGS output: {len(keys2)} rows, {len(c2)} unique keys, {len(keys2)-len(c2)} duplicates")