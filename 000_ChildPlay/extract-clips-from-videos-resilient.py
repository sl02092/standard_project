# IMPORTS
import argparse
import os
import shlex
import subprocess as sp

import cv2
import pandas as pd
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# PATCHED (2026-07-17) -- adds per-clip error handling and a summary log.
# NOTHING about the original extraction logic below is changed -- the
# SPECIAL_CLIPS list, the downsample skip-read, the frame_start-1 vs -2
# cap.set() offsets, the ffmpeg piping -- all copied verbatim, since that
# logic exists specifically to reproduce a historical extraction exactly
# so images stay aligned with already-existing annotations, and any
# rewrite risks breaking that alignment.
#
# WHY THIS PATCH EXISTS: cv2.VideoCapture on a missing file doesn't raise
# -- it silently returns an unopened capture, and the ORIGINAL script's
# very next lines (`ret, frame = cap.read()` then `frame.shape`) throw an
# unhandled AttributeError as soon as `frame` is None. With ~19-20 of 95
# source videos unavailable (author-confirmed missing/private on
# YouTube), the original script would crash on the first affected clip
# and silently process nothing after that point in clips.csv -- not
# "skip missing ones", just stop. This wraps each clip's processing so a
# missing/corrupt video is logged and skipped, not fatal to the whole run.
# ══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Extract clips from videos')
parser.add_argument('--clip_csv_path', type=str, default='./clips.csv', help='Path to the csv file containing information about the clips.')
parser.add_argument('--video_folder', type=str, default='./videos', help='Path to the folder containing the videos.')
parser.add_argument('--clip_folder', type=str, default='./clips', help='Path to the folder where the clips will be saved.')
parser.add_argument('--image_folder', type=str, default='./images', help='Path to the folder where the clip images will be saved.')
parser.add_argument('--skip_report', type=str, default='./extraction_skip_report.csv',
                     help='Where to write the list of clips that could not be processed, with reasons.')
args = parser.parse_args()

# These clips were initially extracted with an incorrect lag of 1 frame. We maintain the same lag here to ensure the frames match the annotations
SPECIAL_CLIPS = ["smwfiZd8HLc_7508-8408-downsampled", "smwfiZd8HLc_10346-12974-downsampled", 
                 "smwfiZd8HLc_16176-16586-downsampled", "smwfiZd8HLc_20322-20668-downsampled", 
                 "31lG75MDwSA_2857-2928", "31lG75MDwSA_3180-3306", "31lG75MDwSA_4686-4764", 
                 "EuYseDl2jm8_1869-2033", "EuYseDl2jm8_3680-3791"]


def process_one_clip(row):
    """Body identical to the original script's loop content, unchanged --
    only extracted into a function so it can be wrapped in a try/except
    per-clip without touching the logic itself."""
    clip_name = row['clip']
    video_name = row['video_id']
    frame_count = row['frame_count']

    clip_image_folder = os.path.join(args.image_folder, clip_name)
    os.makedirs(clip_image_folder, exist_ok=True)

    interval = clip_name.rsplit("_", 1)[1] # "clip-name_start-end-downsampled" -> "start-end-downsampled"
    downsampled = "downsampled" in interval
    interval = interval.replace("-downsampled", "") # "start-end-downsampled" -> "start-end"
    frame_start, frame_end = interval.split("-")
    frame_start, frame_end = int(frame_start), int(frame_end)

    video_path = os.path.join(args.video_folder, f'{video_name}.mp4')
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"source video not found: {video_path}")

    # Read original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cv2 could not open (unreadable/corrupt?): {video_path}")

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = 30 if downsampled else int(round(cap.get(cv2.CAP_PROP_FPS)))

    # Read first frame in the video and retrieve attributes
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise IOError(f"could not read first frame from: {video_path}")
    height, width, _ = frame.shape

    # Set the video to frame start
    if clip_name in SPECIAL_CLIPS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 2)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)

    # Create ffmpeg process
    output_clip_file = os.path.join(args.clip_folder, f"{clip_name}.mp4")
    command = f'ffmpeg -loglevel error -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_clip_file}'
    process = sp.Popen(shlex.split(command), stdin=sp.PIPE)

    clip_frame_num = 0
    ret, frame = cap.read()
    while ret:
        clip_frame_num += 1
        if downsampled: # skip every other frame when FPS is 60
            ret, frame = cap.read()

        # Write frame
        process.stdin.write(frame.tobytes())
        output_frame_file = os.path.join(clip_image_folder, f"{video_name}_{frame_start + clip_frame_num - 1}.jpg")
        cv2.imwrite(output_frame_file, frame)

        ret, frame = cap.read()

        if clip_frame_num == frame_count:
            break

    # Release Capture Device
    cap.release()

    # Close and flush stdin
    process.stdin.close()
    # Wait for sub-process to finish
    process.wait()
    # Terminate the sub-process
    process.terminate()

    return clip_frame_num, frame_count


def main():
    ## Create output clip folder if it doesn't exist
    if not os.path.exists(args.clip_folder):
        os.makedirs(args.clip_folder)

    ## Load CSV file
    df_clips = pd.read_csv(args.clip_csv_path)
    num_clips = len(df_clips)
    print(f"Found {num_clips} clips in the CSV file.")

    n_ok = 0
    skipped = []  # (clip_name, video_name, reason)

    for ix, row in tqdm(df_clips.iterrows(), total=num_clips):
        clip_name = row['clip']
        video_name = row['video_id']
        try:
            written, expected = process_one_clip(row)
            if written != expected:
                # Not fatal -- video existed and was readable, but ran out
                # of frames before reaching frame_count (e.g. a video that
                # was itself truncated). Logged, not silently accepted.
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
        # Quick summary of which source videos are actually the problem,
        # not just how many clips were affected -- a handful of missing
        # videos can account for many skipped clips.
        missing_videos = sorted(set(v for _, v, r in skipped if "not found" in r))
        if missing_videos:
            print(f"{len(missing_videos)} distinct source video(s) missing/unreadable:")
            for v in missing_videos:
                print(f"  {v}")
    else:
        print("No clips skipped.")


if __name__ == '__main__':
    main()
