import pandas as pd
df = pd.read_csv("frame_manifest.csv")
print(df["use_teacher"].value_counts())
print(df["gaze_type"].value_counts())
print("\nCross-tab:")
print(pd.crosstab(df["gaze_type"], df["use_teacher"]))