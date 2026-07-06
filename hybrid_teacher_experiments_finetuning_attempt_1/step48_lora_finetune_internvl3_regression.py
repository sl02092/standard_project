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
MANIFEST_PATH = "frame_manifest.csv"
INTERNVL_NAME = "OpenGVLab/InternVL3-8B-hf"

# PLACEHOLDER -- replace with VAT's actual official train/val split before
# trusting any reported number. Currently just a guess to get the pipeline
# running; splitting by SHOW (not frame) to avoid near-duplicate leakage
# between train and val from the same clip.
TRAIN_SHOWS = ["My Dinner with Andre", "Gone with the Wind", "Tartuffe"]
VAL_SHOWS   = ["Conan"]

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

        gaze_x, gaze_y = float(row["gaze_x"]), float(row["gaze_y"])
        if gaze_x != -1 and gaze_y != -1:
            norm_x, norm_y = gaze_x / w, gaze_y / h
            is_offscreen = 0.0
        else:
            norm_x, norm_y = 0.5, 0.5  # placeholder, masked out of coord loss
            is_offscreen = 1.0

        return {
            "image": img,
            "target_xy": torch.tensor([norm_x, norm_y], dtype=torch.float32),
            "is_offscreen": torch.tensor([is_offscreen], dtype=torch.float32),
        }


def load_split_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    # GT-labelled frames only. Adjust the column/condition here if your
    # manifest marks GT rows differently (e.g. a 'label_source' column
    # from the teacher pipeline's own output, rather than the raw
    # manifest's 'use_teacher' flag).
    if "label_source" in df.columns:
        df = df[df["label_source"] == "gt"]
    else:
        df = df[df["use_teacher"] == False]

    train_df = df[df["show"].isin(TRAIN_SHOWS)].reset_index(drop=True)
    val_df   = df[df["show"].isin(VAL_SHOWS)].reset_index(drop=True)

    if TEST_MODE:
        train_df = train_df.head(30)
        val_df = val_df.head(10)

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

    outputs = internvl_model(**inputs, output_hidden_states=True)
    # Last layer, last token position -- the point at which the model has
    # attended over the full image+prompt context. Same pooling choice as
    # train_v5_regression.py; flagged in the module docstring as worth
    # revisiting if results look degenerate.
    last_hidden = outputs.hidden_states[-1]
    pooled = last_hidden[:, -1, :]
    return pooled


def evaluate(internvl_model, internvl_processor, reg_head, val_df, base_path, device, tag="eval"):
    """Returns mean ADE on val_df. Used BOTH for the pre-training zero-shot
    baseline and for post-training comparison, on the SAME held-out set."""
    dists = []
    reg_head.eval()
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img = Image.open(row["img_path"]).convert("RGB")
            w, h = img.size
            gaze_x, gaze_y = float(row["gaze_x"]), float(row["gaze_y"])
            if gaze_x == -1 and gaze_y == -1:
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

    mean_ade = sum(dists) / len(dists) if dists else float("nan")
    print(f"[{tag}] mean ADE over {len(dists)} on-screen val frames: {mean_ade:.4f}")
    reg_head.train()
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
        trust_remote_code=True, device_map="auto")
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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
