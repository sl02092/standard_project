import csv
from collections import defaultdict

key_splits = defaultdict(set)
with open('frame_manifest_ucolaeo.csv') as f:
    for row in csv.DictReader(f):
        key_splits[(row['show'], row['fname'], row['subject'])].add(row['split'])

cross_split = {k: v for k, v in key_splits.items() if len(v) > 1}
same_split = {k: v for k, v in key_splits.items() if len(v) == 1}
print(f"Duplicate keys spanning multiple splits: {len(cross_split)}")
print(f"Duplicate keys within the same split: {sum(1 for k in same_split if list(same_split[k]) and True) - len(key_splits) + len(cross_split) + len(same_split)}")
# simpler: just show a few examples of each
print("Sample cross-split duplicates:", list(cross_split.items())[:5])