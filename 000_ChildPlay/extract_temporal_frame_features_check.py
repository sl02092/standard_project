import json
from collections import OrderedDict

unique = OrderedDict()
with open("childplay_temporal_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if not row["window_valid"]:
            continue
        clip, subject = row["clip_id"], row["subject_id"]
        for path in row["context_frame_paths"]:
            key = f"{clip}/{subject}/{path.split(chr(92))[-1].split('/')[-1]}"
            unique[key] = True
print(len(unique))