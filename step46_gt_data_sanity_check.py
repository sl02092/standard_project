import pandas as pd
df = pd.read_csv("frame_manifest.csv")
offscreen = df[df["use_teacher"] == False]
print(offscreen[["show", "clip", "fname", "gaze_type"]].head(10))
print(f"\nTotal GT rows: {len(offscreen)}")
print(f"Shows with GT rows: {offscreen['show'].nunique()}")