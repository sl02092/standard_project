"""
extract_frames_vacation.py — extracts frames from raw VACATION source
videos, one folder per video.

WHY THIS IS SIMPLER THAN CHILDPLAY'S EXTRACTION: no historical extraction
to reproduce here (no SPECIAL_CLIPS-style lag correction, no clip-level
start/end range to seek to). VACATION's annotation files are per-VIDEO,
not per-clip, and the frame column is a straightforward 0-indexed
sequential count from that video's own start (confirmed directly against
NewAnt_106.txt: first row is frame 0, max observed frame is 63 -- exactly
matches reading a video from its own beginning, no offset needed).

FRAME COUNT: VACATION has no clips.csv-equivalent giving an expected
frame count per video. Rather than requiring one, this reads each video's
OWN annotation file first and extracts frames 0 through max(frame
column) -- self-bounding, avoids over-extracting frames with no
annotation to ever use, and needs no extra config.

VIDEO LIST: derived by globbing the annotations folder for
"NewAnt_*.txt", not by parsing readme.txt's train/val/test lists --
these are ASSUMPTIONS, not yet confirmed:
  1. annotation files are named "NewAnt_{video_id}.txt", one per video,
     all in one flat folder
  2. source videos are named "{video_id}.mp4" (extension guessed,
     override via --video_ext if wrong)
Both need confirming against the real HPC layout before trusting this at
scale -- correct filenames for the two example files sent so far
(NewAnt_106.txt, NewAnt_170.txt), but only two data points.

NAMING: {image_folder}/{video_id}/{video_id}_{frame}.jpg -- new
convention, no existing precedent to match since VACATION has never been
extracted before. Matches VAT/ChildPlay's "{id}_{frame}.jpg" pattern for
downstream consistency, without ChildPlay's clip-relative offset (not
applicable here -- no clips).

RESILIENCE: same per-video try/except + skip-report pattern as
extract-clips-from-videos-resilient.py, since there's no reason to assume
VACATION's source videos are any less likely to have missing/corrupt
files than ChildPlay's were.
"""

import argparse
import glob
import os

import cv2
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract frames from raw VACATION videos')
parser.add_argument('--annotations_folder', type=str, default='./annotations',
                     help='Folder containing NewAnt_{video_id}.txt files.')
parser.add_argument('--video_folder', type=str, default='./videos',
                     help='Folder containing the raw source videos.')
parser.add_argument('--video_ext', type=str, default='mp4',
                     help='Source video file extension (no dot).')
parser.add_argument('--image_folder', type=str, default='./images',
                     help='Folder where extracted frames will be saved.')
parser.add_argument('--skip_report', type=str, default='./vacation_extraction_skip_report.csv',
                     help='Where to write the list of videos that could not be processed, with reasons.')
args = parser.parse_args()


def get_max_frame(ann_path):
    """Scans an annotation file's frame column (index 5, 0-indexed) and
    returns the maximum frame number referenced. Skips malformed/short
    lines rather than crashing on them -- object rows and person rows
    both have the frame column at the same position, so no row-type
    branching is needed just for this."""
    max_frame = -1
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                frame = int(parts[5])
            except ValueError:
                continue
            max_frame = max(max_frame, frame)
    return max_frame


def process_one_video(video_id):
    ann_path = os.path.join(args.annotations_folder, f"NewAnt_{video_id}.txt")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"annotation file not found: {ann_path}")

    max_frame = get_max_frame(ann_path)
    if max_frame < 0:
        raise ValueError(f"no valid frame rows found in {ann_path}")
    n_frames_needed = max_frame + 1  # frames 0..max_frame inclusive

    video_path = os.path.join(args.video_folder, f"{video_id}.{args.video_ext}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"source video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cv2 could not open (unreadable/corrupt?): {video_path}")

    image_dir = os.path.join(args.image_folder, video_id)
    os.makedirs(image_dir, exist_ok=True)

    n_written = 0
    ret, frame = cap.read()
    while ret and n_written < n_frames_needed:
        out_path = os.path.join(image_dir, f"{video_id}_{n_written}.jpg")
        cv2.imwrite(out_path, frame)
        n_written += 1
        ret, frame = cap.read()

    cap.release()
    return n_written, n_frames_needed


def main():
    ann_files = sorted(glob.glob(os.path.join(args.annotations_folder, "NewAnt_*.txt")))
    video_ids = [os.path.basename(p)[len("NewAnt_"):-len(".txt")] for p in ann_files]
    print(f"Found {len(video_ids)} annotation files in {args.annotations_folder}")

    if not video_ids:
        print("ERROR: no NewAnt_*.txt files found -- check --annotations_folder "
              "and confirm the naming convention before rerunning.")
        return

    n_ok = 0
    skipped = []

    for video_id in tqdm(video_ids):
        try:
            written, expected = process_one_video(video_id)
            if written != expected:
                skipped.append((video_id, f"partial: wrote {written}/{expected} frames"))
            else:
                n_ok += 1
        except Exception as e:
            skipped.append((video_id, f"{type(e).__name__}: {e}"))
            continue

    print(f"\n{n_ok}/{len(video_ids)} videos fully extracted.")
    if skipped:
        print(f"{len(skipped)} videos skipped or partial -- see {args.skip_report}")
        pd.DataFrame(skipped, columns=["video_id", "reason"]).to_csv(args.skip_report, index=False)
        missing_videos = [v for v, r in skipped if "source video not found" in r]
        if missing_videos:
            print(f"{len(missing_videos)} video(s) missing/unreadable:")
            for v in missing_videos:
                print(f"  {v}")
    else:
        print("No videos skipped.")


if __name__ == '__main__':
    main()
