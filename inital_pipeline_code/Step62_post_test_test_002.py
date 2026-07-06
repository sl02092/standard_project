import json, math
records = [json.loads(l) for l in open("labels_test.jsonl") if l.strip()]
dists = [math.sqrt((r["pred_x"]-r["gt_x"])**2 + (r["pred_y"]-r["gt_y"])**2)
         for r in records if r.get("pred_x") and r.get("gt_x")]
print(f"Mean: {sum(dists)/len(dists):.4f}  Max: {max(dists):.4f}  Min: {min(dists):.4f}")

'''
Mean: 0.1263  Max: 0.7472  Min: 0.0027
'''