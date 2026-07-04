import json

object_rows = []
with open("labels_full.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row.get("gaze_type") == "object" and row.get("label_source") == "teacher":
            object_rows.append(row)

print(f"Object-gaze, teacher-labelled rows: {len(object_rows)}")

# Mean distance error specifically on object frames
import math
dists = [math.hypot(r["pred_x"]-r["gt_x"], r["pred_y"]-r["gt_y"]) for r in object_rows]
print(f"Mean distance error on object frames: {sum(dists)/len(dists):.3f}")

# Print a handful to eyeball raw_response content
for r in object_rows[:5]:
    print(r["gt_x"], r["gt_y"], "→", r["pred_x"], r["pred_y"])
    print(" ", r["raw_response"][:150])
    print()