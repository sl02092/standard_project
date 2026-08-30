import pandas as pd
df = pd.read_csv('frame_manifest_childplay.csv', dtype={'subject': str})
n_subj = df.groupby(['show','clip','fname'])['subject'].transform('nunique')
obj = df[df['gaze_type']=='object']
solo = (n_subj[obj.index] <= 1).sum()
print(f"{solo} / {len(obj)} object frames ({100*solo/len(obj):.1f}%) are solo — step40 will GT-fallback these")