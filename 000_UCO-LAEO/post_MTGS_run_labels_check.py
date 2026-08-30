import json
from collections import Counter

keys = []
with open('labels_ucolaeo_full.jsonl') as f:
    for line in f:
        r = json.loads(line)
        keys.append((r['show'], r['fname'], r['subject']))

c = Counter(keys)
print(f"Total rows: {len(keys)}, unique keys: {len(c)}, duplicates: {len(keys) - len(c)}")