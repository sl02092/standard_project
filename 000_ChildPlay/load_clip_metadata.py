import os
from PIL import Image
from childplay_annotation_utils import load_clip_metadata, resolve_image_path

meta = load_clip_metadata()
checked = 0
for clip_id, info in meta.items():
    if info["downsampled"]:
        continue
    img_path = resolve_image_path(clip_id, 1)
    if not os.path.isfile(img_path):
        continue
    with Image.open(img_path) as img:
        actual = img.size
    print(f"{clip_id:40s} csv_resolution={info['resolution']:<12} actual={actual}")
    checked += 1
    if checked >= 15:
        break