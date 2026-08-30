import json
from collections import defaultdict

rows_by_key = defaultdict(list)
with open('labels_ucolaeo_hybrid_targeted.jsonl') as f:
    for line in f:
        r = json.loads(line)
        rows_by_key[(r['show'], r['fname'], r['subject'])].append(r)

disagreements = 0
for key, rows in rows_by_key.items():
    if len(rows) > 1:
        preds = {(round(r['pred_x'], 4), round(r['pred_y'], 4)) for r in rows}
        if len(preds) > 1:
            disagreements += 1

print(f"Duplicate keys with disagreeing predictions: {disagreements}")