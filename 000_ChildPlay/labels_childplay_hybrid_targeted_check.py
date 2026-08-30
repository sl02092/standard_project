import json
from collections import Counter

c = Counter()
with open('labels_childplay_hybrid_targeted.jsonl') as f:
    for line in f:
        r = json.loads(line)
        c[r.get('label_source')] += 1

print(c)