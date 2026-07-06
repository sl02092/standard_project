"""
step48_lora_finetune_internvl3_regression.py — FIRST-PASS DRAFT, UNTESTED
AGAINST REAL WEIGHTS. Written without GPU/model access; expect to debug on
first run. TEST_MODE=True by default for exactly this reason -- run small
before committing to a long job.

GOAL: test whether LoRA-adapting InternVL3-8B-hf's OWN native architecture
(its own vision encoder + its own existing projector, both left frozen)
closes some of the remaining object-gaze ADE gap, as a coordinate-
regression head reads out the model's own hidden state.

RELATIONSHIP TO train_step43d8_train_v5_regression.py:
  That script bypassed InternVL3's own vision pipeline entirely -- DINOv2
  features were pushed through a brand-new, randomly-initialized projector
  straight into the frozen LLM, discarding InternVL3's own pretrained
  cross-modal alignment. Closed negative (ADE 0.554, memorization-pattern
  outputs), plausibly because a from-scratch projector can't learn a full
  vision-language bridge from 5000 samples.
  This script does NOT repeat that design. It uses InternVL3-8B-hf's own
  processor/vision-encoder/projector pipeline unchanged (same as the
  production teacher pipeline), and only adds LoRA adapters to the
  LANGUAGE MODEL's attention projections -- a much smaller, better-
  conditioned intervention on top of an already-aligned model, rather than
  rebuilding the cross-modal bridge from zero.

WHAT THIS DOES:
  1. Loads InternVL3-8B-hf normally (same as step46/step47).
  2. Wraps it with LoRA adapters (peft) on the language model's attention
     projections ONLY. Vision encoder + existing vision-language projector
     stay FROZEN -- unchanged from the pretrained model.
  3. Adds a small regression head (same SmoothL1 + BCE-offscreen design as
     train_v5_regression.py, for methodological consistency) reading the
     LAST hidden state at the last token position, exactly as step46/
     step47 read out text -- except this reads a continuous embedding, not
     generated tokens.
  4. Trains ONLY on GT-labelled frames (label_source == "gt" in your
     manifest/labels), never on teacher/pseudo-labels -- avoids the
     circularity of fine-tuning against the very labels this teacher
     itself produces.
  5. Uses a genuine held-out split (by SHOW, not by frame, to avoid
     leaking near-duplicate frames from the same clip across train/val)
     and reports a ZERO-SHOT baseline ADE on that same held-out val set
     BEFORE training starts, so any improvement is measured against a
     fair, matched baseline -- the previous attempt didn't do this.

THINGS THAT WILL LIKELY NEED FIXING ON FIRST REAL RUN (flagged honestly,
not discovered by testing, since this was written without model access):
  - LORA_TARGET_MODULES below is a best guess (Qwen2-style attention
    projection names, since InternVL3-8B commonly pairs with a Qwen2.5
    LLM backbone). If this is wrong for the actual loaded architecture,
    peft will raise a clear "no matching modules" error on
    get_peft_model() -- run print_module_names() (below) FIRST if that
    happens, then fix LORA_TARGET_MODULES and rerun. Don't guess twice;
    inspect once.
  - TRAIN_SHOWS / VAL_SHOWS below are PLACEHOLDERS. You mentioned VAT
    already has an official train/test split -- replace these with the
    actual split (by show or by whatever unit VAT's own split uses)
    before trusting any number this script reports. Training against the
    wrong split silently gives you a meaningless-but-plausible-looking ADE.
  - GT-frame count for fine-tuning may be small once restricted to a
    proper train split -- if too few object-gaze GT frames exist to
    learn anything, that itself is a useful (negative) finding, not a bug.
  - Pooling strategy (last-token hidden state) mirrors train_v5_regression
    for consistency, but is itself an assumption worth questioning if
    results look degenerate (e.g. all predictions collapsing to the same
    point, as happened before) -- mean-pooling over the vision-token span
    specifically (rather than the final text token) is a reasonable next
    thing to try if that happens.

OUTPUT: LoRA adapter weights (peft's own save format, small) +
regression head state dict, saved separately so a partial/failed run
never overwrites the base model or an earlier good checkpoint.
"""

import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError(
        "peft is required for this script. Install with: "
        "pip install peft --break-system-packages"
    )

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

TEST_MODE = True   # RUN THIS FIRST. ~30 GT frames, 1 epoch, no checkpointing
                    # noise -- confirms the whole pipeline runs end-to-end
                    # before you trust it with a real job. Flip to False
                    # only after a clean TEST_MODE pass.

BASE_PATH     = os.environ.get(
    "VAT_BASE_PATH",
    "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget")
# CORRECTED (again): the raw frame_manifest.csv, not a teacher-pipeline
# output file. label_source == "gt" in labels_full.jsonl etc. does NOT
# mean "genuine on-screen ground truth" -- per the pipeline's own routing
# convention (use_teacher == False -> GT direct), "gt" specifically means
# OFF-SCREEN frames routed straight to ground truth because there's
# nothing to reason about. Every val "gt_x"/"gt_y" from that file was
# ~0.000 -- the mis-normalized -1 sentinel (-1/img_w), not a real
# coordinate -- which is what the trivial-baseline sanity check caught.
# frame_manifest.csv has genuine, independent, human-annotated gaze_x/
# gaze_y for EVERY frame (on-screen and off-screen alike), completely
# untouched by any VLM inference -- no circularity concern, and the
# correct on-screen filter is simply gaze_x != -1, nothing to do with
# label_source or use_teacher.
MANIFEST_PATH = os.environ.get("LORA_LABELS_PATH", "frame_manifest.csv")
INTERNVL_NAME = "OpenGVLab/InternVL3-8B-hf"

# REAL VAT official split, confirmed via `diff` against the actual
# annotations/train and annotations/test directories on Eureka2 -- 39
# train shows, 10 test shows, no overlap.
#
# The official TEST split (CBS This Morning, Downton Abby, Hell's Kitchen,
# It's Always Sunny in Philadelphia, I Wanna Marry Harry, Jamie Oliver,
# MLB Interview, Survivor, Titanic, West World) is DELIBERATELY NOT used
# anywhere below -- reserved untouched for a final, trustworthy number
# later, not spent on exploratory LoRA tuning.
#
# Instead, an INTERNAL validation set is carved out of the 39 TRAIN shows,
# covering multiple genres (sitcom, drama, talk show, film) so a result
# can't be an artifact of one show's narrow gaze-target distribution --
# exactly the problem the previous single-show ("Conan"-only) val set had.
_ALL_OFFICIAL_TRAIN_SHOWS = [
    "All in the Family", "A Play With Words", "Arrested Development",
    "Band of Brothers", "Before Sunrise", "Big Bang Theory", "Breaking Bad",
    "BTS at Jimmy Fallon", "Cheers", "Conan", "Coveted", "Crazy Rich Asian",
    "Driving Miss Daisy", "Friends", "Give Me One Reason",
    "Gone with the Wind", "Grey's Anatomy", "Hearing",
    "How I Met Your Mother", "Interview at the Oscars",
    "Interview with Bill Gates", "Jersey Shore",
    "Keeping Up With the Kardashians", "Modern Family",
    "My Dinner with Andre", "Orange is the New Black", "Project Runway",
    "Secret", "Seinfeld", "Sherlock", "Silicon Valley", "Sound of Music",
    "Star Wars", "Suits", "Tartuffe", "The Ellen Show", "The View",
    "Three Idiots", "UFC Octagon Interview", "Veep",
]

VAL_SHOWS = ["Conan", "Big Bang Theory", "Friends", "Modern Family",
             "My Dinner with Andre", "Sherlock"]
TRAIN_SHOWS = [s for s in _ALL_OFFICIAL_TRAIN_SHOWS if s not in VAL_SHOWS]

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
# BEST GUESS for a Qwen2.5-style LLM backbone -- VERIFY against the actual
# loaded model before trusting this. See print_module_names() below.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 2e-4   # typical LoRA LR, ~10x a full-finetune LR; not tuned
WEIGHT_DECAY  = 0.01
EPOCHS        = 3 if not TEST_MODE else 1
BATCH_SIZE    = 1      # per-sample forward/backward, same constraint as
                        # train_v5_regression.py (no batching across the
                        # LLM's variable-length multimodal sequences here)
SAVE_EVERY_N_STEPS = 100

OUTPUT_DIR = "lora_regression_test" if TEST_MODE else "lora_regression_run"

IDENTIFY_PROMPT_FOR_TRAINING = (
    "Trace the line of sight vector from the eyes of the subject. "
    "Identify the precise (x, y) location they are looking at."
)


# ══════════════════════════════════════════════════════════════════════
# ── DIAGNOSTIC: run this FIRST if LoRA target modules don't match ─────
# ══════════════════════════════════════════════════════════════════════

def print_module_names(model, max_lines=200):
    """
    If get_peft_model() raises a 'target modules not found' error, run:
        print_module_names(internvl_model.language_model)
    to see the ACTUAL attention projection names in this architecture,
    then fix LORA_TARGET_MODULES above. Don't guess a second time.
    """
    seen = 0
    for name, _ in model.named_modules():
        if any(key in name for key in ("proj", "attn", "attention")):
            print(name)
            seen += 1
            if seen >= max_lines:
                print("... (truncated)")
                break


# ══════════════════════════════════════════════════════════════════════
# ── REGRESSION HEAD (same design as train_v5_regression.py, for
#    methodological consistency -- SmoothL1 coords + separate offscreen
#    logit, NOT a sentinel value crammed into the coordinate space) ────
# ══════════════════════════════════════════════════════════════════════

class GazeCoordRegressionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 256), nn.GELU())
        self.coord_out = nn.Linear(256, 2)
        self.offscreen_out = nn.Linear(256, 1)
        for m in [self.net[0], self.coord_out, self.offscreen_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, pooled_hidden):
        feat = self.net(pooled_hidden.float())  # float32 for head stability
        coords = torch.sigmoid(self.coord_out(feat))
        offscreen_logit = self.offscreen_out(feat)
        return coords, offscreen_logit


# ══════════════════════════════════════════════════════════════════════
# ── DATA ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class GTFrameDataset(Dataset):
    """
    GT-LABELLED FRAMES ONLY -- never teacher/pseudo-labels, to avoid
    fine-tuning the teacher against its own (possibly wrong) output.
    """
    def __init__(self, manifest_df, base_path):
        self.df = manifest_df.reset_index(drop=True)
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        w, h = img.size

        # frame_manifest.csv stores RAW PIXEL gaze_x/gaze_y -- normalize
        # here. Already filtered to on-screen only in load_split_manifest,
        # but the is_offscreen path is kept as a defensive fallback.
        gaze_x, gaze_y = float(row["gaze_x"]), float(row["gaze_y"])
        if gaze_x != -1 and gaze_y != -1:
            norm_x, norm_y = gaze_x / w, gaze_y / h
            is_offscreen = 0.0
        else:
            norm_x, norm_y = 0.5, 0.5
            is_offscreen = 1.0

        return {
            "image": img,
            "target_xy": torch.tensor([norm_x, norm_y], dtype=torch.float32),
            "is_offscreen": torch.tensor([is_offscreen], dtype=torch.float32),
        }


def gaze_collate_fn(batch):
    """
    Custom collate function -- REQUIRED because default_collate has no idea
    how to batch raw PIL.Image objects (it only knows tensors, numbers,
    dicts, lists natively). Images are left as a plain Python list; the
    numeric fields are tensor-stacked normally. main() already expects
    batch["image"] to be a list (it does batch["image"][0]), so no
    downstream change is needed beyond wiring this into the DataLoader.
    """
    return {
        "image": [item["image"] for item in batch],
        "target_xy": torch.stack([item["target_xy"] for item in batch]),
        "is_offscreen": torch.stack([item["is_offscreen"] for item in batch]),
    }


def load_split_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"'{MANIFEST_PATH}' not found in the current working directory "
            f"({os.getcwd()}). Set LORA_LABELS_PATH if it's elsewhere."
        )

    # NEW: pandas' own CSV tokenizer (BOTH engines) fails on this file with
    # a field-count error that THREE independent checks (awk, iconv, and
    # Python's stdlib csv module) all disagree with -- the file itself is
    # confirmed fine (valid UTF-8, no malformed rows by a quote-aware
    # parser). Rather than keep guessing at pandas' internal tokenizer
    # quirk, use the parser already PROVEN to read this file correctly --
    # stdlib csv -- and only hand pandas the already-parsed rows.
    import csv as csv_module
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv_module.reader(f)
        header = next(reader)
        rows = []
        bad_rows_shown = 0
        bad_row_count = 0
        for i, row in enumerate(reader, start=2):  # start=2: line 1 is header
            if len(row) != len(header):
                bad_row_count += 1
                if bad_rows_shown < 5:
                    print(f"[!] Line ~{i}: expected {len(header)} fields, "
                          f"got {len(row)} -- raw content: {row}")
                    bad_rows_shown += 1
                continue  # skip rather than crash -- see printed evidence above
            rows.append(dict(zip(header, row)))

    if bad_row_count:
        print(f"[!] Skipped {bad_row_count} malformed rows out of "
              f"{bad_row_count + len(rows)} total data rows "
              f"(showing first {bad_rows_shown} above).")

    df = pd.DataFrame(rows)
    print(f"[+] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    if "gaze_x" not in df.columns or "gaze_y" not in df.columns:
        raise KeyError(
            f"'gaze_x'/'gaze_y' still missing after parsing. Actual columns "
            f"found: {list(df.columns)}. Check the [!] diagnostic output "
            f"above for malformed-row evidence."
        )

    # csv.reader returns everything as strings -- explicitly cast the
    # numeric column this script actually depends on. Skipping this would
    # silently break the on-screen filter below: "-1" (string) != -1 (int)
    # is always True, which would make EVERY row look on-screen.
    df["gaze_x"] = df["gaze_x"].astype(float)
    df["gaze_y"] = df["gaze_y"].astype(float)

    # Genuine on-screen frames, straight from the raw human annotation --
    # NOT filtered via label_source or use_teacher (see MANIFEST_PATH
    # comment above for why that was wrong). This is the real, independent
    # ground truth: never touched by the VLM teacher, so no circularity
    # concern fine-tuning against it.
    before = len(df)
    df = df[(df["gaze_x"] != -1) & (df["gaze_y"] != -1)]
    print(f"[+] {len(df)} genuine on-screen frames found in {MANIFEST_PATH} "
          f"(of {before} total)")

    train_df = df[df["show"].isin(TRAIN_SHOWS)].reset_index(drop=True)
    val_df   = df[df["show"].isin(VAL_SHOWS)].reset_index(drop=True)

    if TEST_MODE:
        # Sample evenly ACROSS SHOWS, not a blind head() -- a naive head()
        # could still collapse val back down to effectively one show if it
        # has enough GT frames to fill the quota alone, defeating the
        # point of a multi-show val set. Same per-show sampling approach
        # already used in step46/step47's load_manifest().
        train_df = train_df.groupby("show", group_keys=False).head(5).reset_index(drop=True)
        val_df = val_df.groupby("show", group_keys=False).head(3).reset_index(drop=True)

    print(f"[+] GT frames -- train: {len(train_df)}, val: {len(val_df)}")
    if len(train_df) < 20:
        print("[!] WARNING: very few training frames. If this isn't "
              "TEST_MODE, check TRAIN_SHOWS / the GT filter above -- this "
              "looks too small to learn anything meaningful from.")
    return train_df, val_df


# ══════════════════════════════════════════════════════════════════════
# ── MODEL FORWARD PASS (uses InternVL3's OWN vision pipeline via its
#    own processor/chat template -- NOT a hand-spliced embedding, unlike
#    train_v5_regression.py) ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def get_pooled_hidden_state(internvl_model, internvl_processor, img, prompt, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    inputs = internvl_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    outputs = internvl_model(**inputs, output_hidden_states=True, logits_to_keep=1)
    # Last layer, last token position -- the point at which the model has
    # attended over the full image+prompt context. Same pooling choice as
    # train_v5_regression.py; flagged in the module docstring as worth
    # revisiting if results look degenerate.
    last_hidden = outputs.hidden_states[-1]
    pooled = last_hidden[:, -1, :]
    return pooled


def compute_trivial_mean_baseline_ade(train_df, val_df):
    """
    SANITY CHECK: predicts the TRAINING set's mean (x, y) for every val
    frame, completely ignoring image content. If this trivial, content-
    blind baseline lands anywhere near the model's actual ADE, that's
    conclusive evidence the number reflects a low-variance val set (or a
    positional shortcut), not genuine per-frame spatial grounding -- run
    this BEFORE trusting any model result.
    """
    def normalized_xy(row):
        img = Image.open(row["img_path"])
        w, h = img.size
        return float(row["gaze_x"]) / w, float(row["gaze_y"]) / h

    train_on_screen = train_df[(train_df["gaze_x"] != -1) & (train_df["gaze_y"] != -1)]
    train_xy = [normalized_xy(row) for _, row in train_on_screen.iterrows()]
    mean_x = sum(p[0] for p in train_xy) / len(train_xy)
    mean_y = sum(p[1] for p in train_xy) / len(train_xy)

    dists = []
    for _, row in val_df.iterrows():
        if row["gaze_x"] == -1 or row["gaze_y"] == -1:
            continue
        gt_x, gt_y = normalized_xy(row)
        dist = ((gt_x - mean_x) ** 2 + (gt_y - mean_y) ** 2) ** 0.5
        dists.append(dist)

    mean_ade = sum(dists) / len(dists) if dists else float("nan")
    print(f"[trivial-baseline] predicting train-mean ({mean_x:.3f}, {mean_y:.3f}) "
          f"for every val frame -> ADE {mean_ade:.4f} over {len(dists)} frames")
    print("  (if this is anywhere close to the model's ADE below, the model")
    print("   result is NOT trustworthy -- it likely reflects val-set")
    print("   positional bias, not genuine per-frame spatial grounding)")
    return mean_ade


def evaluate(internvl_model, internvl_processor, reg_head, val_df, base_path, device, tag="eval"):
    """Returns mean ADE on val_df. Used BOTH for the pre-training zero-shot
    baseline and for post-training comparison, on the SAME held-out set."""
    dists = []
    raw_predictions = []  # NEW: for manual inspection -- see printout below
    was_training = internvl_model.training
    internvl_model.eval()  # no dropout etc. during eval; also no need for
                            # checkpointing here since there's no backward
    reg_head.eval()
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img = Image.open(row["img_path"]).convert("RGB")
            w, h = img.size
            gaze_x, gaze_y = float(row["gaze_x"]), float(row["gaze_y"])
            if gaze_x == -1 or gaze_y == -1:
                continue  # off-screen frames excluded from ADE, same as
                          # every other ADE number in this project
            gt_x, gt_y = gaze_x / w, gaze_y / h

            pooled = get_pooled_hidden_state(
                internvl_model, internvl_processor, img,
                IDENTIFY_PROMPT_FOR_TRAINING, device)
            pred_xy, _ = reg_head(pooled)
            pred_x, pred_y = pred_xy[0, 0].item(), pred_xy[0, 1].item()

            dist = ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2) ** 0.5
            dists.append(dist)
            raw_predictions.append((row.get("show", "?"), gt_x, gt_y, pred_x, pred_y, dist))

    # NEW: print every raw prediction so a clustered/degenerate result is
    # visible directly, not hidden behind a single aggregate ADE number.
    print(f"  [{tag}] raw predictions (show, gt_x, gt_y, pred_x, pred_y, dist):")
    for show, gx, gy, px, py, d in raw_predictions:
        print(f"    {show:<20} gt=({gx:.3f},{gy:.3f})  pred=({px:.3f},{py:.3f})  dist={d:.4f}")

    mean_ade = sum(dists) / len(dists) if dists else float("nan")
    print(f"[{tag}] mean ADE over {len(dists)} on-screen val frames: {mean_ade:.4f}")
    reg_head.train()
    if was_training:
        internvl_model.train()
    return mean_ade


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Mode: {'TEST_MODE' if TEST_MODE else 'FULL'}")

    train_df, val_df = load_split_manifest()

    print("[*] Loading InternVL3-8B-hf...")
    internvl_processor = AutoProcessor.from_pretrained(INTERNVL_NAME, trust_remote_code=True)
    internvl_model = AutoModelForImageTextToText.from_pretrained(
        INTERNVL_NAME, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto", tie_word_embeddings=False)
    device = internvl_model.device

    # ── Freeze everything, then LoRA-adapt only the language model's
    # attention projections. Vision encoder + its existing projector are
    # UNTOUCHED -- this is the key difference from train_v5_regression.py.
    internvl_model.requires_grad_(False)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )

    try:
        internvl_model = get_peft_model(internvl_model, lora_config)
    except ValueError as e:
        print("\n[!] get_peft_model() failed -- LORA_TARGET_MODULES likely "
              "doesn't match this architecture's actual module names.")
        print("[!] Run print_module_names(internvl_model.language_model) "
              "(or on the base model if .language_model doesn't exist) to "
              "find the correct names, fix LORA_TARGET_MODULES above, and "
              "rerun. Not guessing a second time blind.")
        raise e

    internvl_model.print_trainable_parameters()

    # NEW: gradient checkpointing -- backprop still needs to flow through
    # every frozen layer to reach the LoRA adapters, which retains a LOT of
    # intermediate activation memory otherwise. enable_input_require_grads()
    # is needed alongside it because the input embedding layer is frozen;
    # without it, checkpointed layers can silently fail to pass gradients
    # through correctly. Both are standard practice for LoRA + gradient
    # checkpointing together, not tuned/verified against this specific
    # model -- if gradient_checkpointing_enable() isn't found directly on
    # the peft-wrapped model, try internvl_model.get_base_model().
    internvl_model.gradient_checkpointing_enable()
    internvl_model.enable_input_require_grads()

    # NEW: from_pretrained() commonly leaves the model in eval() mode, and
    # HF's gradient-checkpointing implementation typically checks
    # self.training before actually applying torch.utils.checkpoint --
    # if left in eval mode, checkpointing silently does nothing (no error,
    # just the full memory cost as if it were never enabled). This is the
    # most likely cause of the previous OOM happening during forward,
    # before backward even starts.
    internvl_model.train()

    hidden_dim = internvl_model.config.text_config.hidden_size \
        if hasattr(internvl_model.config, "text_config") \
        else internvl_model.config.hidden_size
    reg_head = GazeCoordRegressionHead(hidden_dim).to(device)

    coord_criterion = nn.SmoothL1Loss()
    offscreen_criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in internvl_model.parameters() if p.requires_grad] \
        + list(reg_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    dataset = GTFrameDataset(train_df, BASE_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=gaze_collate_fn)

    print("\n[*] Trivial sanity-check baseline (no model at all):")
    compute_trivial_mean_baseline_ade(train_df, val_df)

    print("\n[*] Zero-shot baseline (before any LoRA training) on held-out val set:")
    baseline_ade = evaluate(internvl_model, internvl_processor, reg_head, val_df, BASE_PATH, device, tag="baseline")

    print(f"\n[*] Starting LoRA fine-tuning over {len(dataloader)} samples x {EPOCHS} epochs...")
    step = 0
    t_start = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            img = batch["image"][0]
            target_xy = batch["target_xy"].to(device)
            is_offscreen = batch["is_offscreen"].to(device)

            pooled = get_pooled_hidden_state(
                internvl_model, internvl_processor, img,
                IDENTIFY_PROMPT_FOR_TRAINING, device)
            pred_xy, pred_offscreen_logit = reg_head(pooled)

            coord_loss = coord_criterion(pred_xy, target_xy)
            offscreen_loss = offscreen_criterion(pred_offscreen_logit, is_offscreen)
            loss = offscreen_loss + coord_loss * (1.0 - is_offscreen[0, 0])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step += 1

            if step % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  epoch {epoch+1} step {step} loss {loss.item():.4f} "
                      f"({elapsed/step:.2f}s/step)")

            if step % SAVE_EVERY_N_STEPS == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"ckpt_step{step}")
                internvl_model.save_pretrained(ckpt_dir)
                torch.save(reg_head.state_dict(), os.path.join(ckpt_dir, "reg_head.pt"))
                print(f"  [ckpt] saved -> {ckpt_dir}")

        print(f"[epoch {epoch+1}] avg loss: {total_loss / max(len(dataloader), 1):.4f}")

    print("\n[*] Post-training ADE on the SAME held-out val set:")
    final_ade = evaluate(internvl_model, internvl_processor, reg_head, val_df, BASE_PATH, device, tag="post-training")

    print(f"\n[+] Baseline ADE:      {baseline_ade:.4f}")
    print(f"[+] Post-training ADE: {final_ade:.4f}")
    print(f"[+] Change:            {baseline_ade - final_ade:+.4f} (positive = improved)")

    final_dir = os.path.join(OUTPUT_DIR, "final")
    internvl_model.save_pretrained(final_dir)
    torch.save(reg_head.state_dict(), os.path.join(final_dir, "reg_head.pt"))
    with open(os.path.join(final_dir, "results.json"), "w") as f:
        json.dump({"baseline_ade": baseline_ade, "final_ade": final_ade,
                   "train_shows": TRAIN_SHOWS, "val_shows": VAL_SHOWS,
                   "n_train": len(train_df), "n_val": len(val_df)}, f, indent=2)
    print(f"[+] Saved to {final_dir}")


if __name__ == "__main__":
    main()
