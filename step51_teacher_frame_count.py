import pandas as pd
df = pd.read_csv("frame_manifest.csv")
print(df["use_teacher"].value_counts())
print(f"Teacher frames : {df[df['use_teacher']==True]['fname'].nunique()}")
print(f"Estimated hrs  : {df['use_teacher'].sum() * 58 / 3600:.0f} hrs")