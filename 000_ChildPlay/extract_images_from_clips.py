"""
extract_images_from_clips.py — Extract per-frame images directly from
already-extracted ChildPlay clips, rather than from the 95 raw source
videos.

WHY THIS EXISTS: the original extract-clips-from-videos.py opens each raw
source video, seeks to the clip's start frame (with a -2 vs -1 offset for
SPECIAL_CLIPS, reproducing a historical extraction lag), and writes each
selected frame to BOTH a new clip file (via ffmpeg) and an image file, in
the same loop iteration, using the SAME frame object for both. That means
the clip file's frame sequence IS the image sequence -- nothing about
which frames get selected differs between the two outputs. All of the
original script's special-casing (SPECIAL_CLIPS offsets, the downsampling
skip-read) exists only to decide which source-video frames make it into
that shared write -- once a frame is inside the clip, reading the clip
back from its own start and taking every frame in order reproduces the
same result, with NONE of that special-casing needing to be replicated
here. This only needs the 401 already-present clips, not the 95 raw
source videos (of which ~19-20 are missing/private).

CAVEAT: clips were saved with lossy H.264 compression (crf 24), so
re-extracted frames won't be bit-identical to a direct source-video
extraction. Not expected to matter for a model operating on small
(224x224) crops, but worth knowing rather than assuming pixel-perfect
equivalence.

NAMING: identical convention to the original script --
{image_folder}/{clip_name}/{video_id}_{frame_start + n - 1}.jpg -- so
childplay_annotation_utils.py's resolve_image_path() needs no changes.
"""

import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract per-frame images from already-extracted clips')
parser.add_argument('--clip_csv_path', type=str, default='./clips.csv', help='Path to the csv file containing information about the clips.')
parser.add_argument('--clip_folder', type=str, default='./clips', help='Path to the folder containing the already-extracted clips.')
parser.add_argument('--image_folder', type=str, default='./images', help='Path to the folder where the clip images will be saved.')
parser.add_argument('--skip_report', type=str, default='./image_extraction_skip_report.csv',
                     help='Where to write the list of clips that could not be processed, with reasons.')
args = parser.parse_args()


def process_one_clip(row):
    clip_name = row['clip']
    video_name = row['video_id']
    frame_count = row['frame_count']

    interval = clip_name.rsplit("_", 1)[1]  # "clip-name_start-end-downsampled" -> "start-end-downsampled"
    interval = interval.replace("-downsampled", "")
    frame_start, frame_end = interval.split("-")
    frame_start = int(frame_start)

    clip_path = os.path.join(args.clip_folder, f"{clip_name}.mp4")
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"clip not found: {clip_path}")

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"cv2 could not open (unreadable/corrupt?): {clip_path}")

    clip_image_folder = os.path.join(args.image_folder, clip_name)
    os.makedirs(clip_image_folder, exist_ok=True)

    n = 0
    ret, frame = cap.read()
    while ret:
        n += 1
        output_frame_file = os.path.join(clip_image_folder, f"{video_name}_{frame_start + n - 1}.jpg")
        cv2.imwrite(output_frame_file, frame)
        ret, frame = cap.read()
        if n == frame_count:
            break

    cap.release()
    return n, frame_count


def main():
    df_clips = pd.read_csv(args.clip_csv_path)
    num_clips = len(df_clips)
    print(f"Found {num_clips} clips in the CSV file.")

    n_ok = 0
    skipped = []

    for _, row in tqdm(df_clips.iterrows(), total=num_clips):
        clip_name = row['clip']
        video_name = row['video_id']
        try:
            written, expected = process_one_clip(row)
            if written != expected:
                skipped.append((clip_name, video_name,
                                 f"partial: wrote {written}/{expected} frames"))
            else:
                n_ok += 1
        except Exception as e:
            skipped.append((clip_name, video_name, f"{type(e).__name__}: {e}"))
            continue

    print(f"\n{n_ok}/{num_clips} clips fully extracted.")
    if skipped:
        print(f"{len(skipped)} clips skipped or partial -- see {args.skip_report}")
        pd.DataFrame(skipped, columns=["clip", "video_id", "reason"]).to_csv(
            args.skip_report, index=False
        )
    else:
        print("No clips skipped.")


if __name__ == '__main__':
    main()
