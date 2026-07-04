"""
train_v5_regression.py — Direct coordinate-regression variant.

Same DINOv2 + InternVL3-8B-hf hybrid architecture as
train_step43d7_hybrid_004_hpc.py, but replaces next-token text-prediction
(CrossEntropyLoss over decimal-string tokens) with direct (x,y) regression
(SmoothL1Loss) plus a separate off-screen classification head
(BCEWithLogitsLoss).

WHY: diagnostic evidence on the text-generation version showed near-zero
parse failures but WORSE spatial ADE after 33x more training data (0.554
vs 0.452 on the same 150-sample model), plus identical predictions on
genuinely different frames (e.g. 2250/2295 and 2250/2300 both -> (0.45,0.25))
-- consistent with cross-entropy over text tokens being a poor, distance-
unaware training signal for a continuous spatial task. This version tests
whether a direct, distance-aware regression loss (the same SmoothL1 already
used by the actual Stage-1 student, for methodological consistency) produces
better spatial grounding from the same data and same frozen backbones.

OUTPUT: saved to dino_internvl_REGRESSION_final.pt (deliberately different
filename from the text-generation version's dino_internvl_projector.pt --
this checkpoint has a different proj_head state AND an additional reg_head
that the old eval/diagnostic script cannot load).
"""

import os
import re
import torch
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# ── CONFIGURATION & DEFAULT HYPERPARAMETERS ───────────────────────────
BASE_PATH         = r"C:\repo\standard_project\videoattentiontarget"
MANIFEST_PATH     = r"C:\repo\standard_project\frame_manifest.csv"
DINO_MODEL_NAME   = "facebook/dinov2-large"
LLM_MODEL_NAME    = "OpenGVLab/InternVL3-8B-hf"

# Scalability Parameters (Set NUM_TRAIN_SAMPLES = -1 to consume the entire manifest)
# CAPPED at 5000 for this run: at ~0.94 sec/frame for a single InternVL3-8B forward
# pass (measured from the production teacher pipeline, 23,096 frames / 6hrs), a
# full-manifest x 3-epoch run here would mean THREE PASSES over all 23,096 frames,
# each sample requiring an uncached forward+backward through the LLM (no batching
# benefit -- see the per-sample loop below). Rough estimate: 54-90 hours for the
# full manifest at 3 epochs, well over the agreed 24hr exploratory budget. 5000
# samples keeps real margin under even the conservative estimate, while being
# ~33x more data than the original 150-sample local test.
NUM_TRAIN_SAMPLES = 5000
BATCH_SIZE        = 4     # Scale up on the HPC (e.g., 32 for L40S, 64 for A100)
LEARNING_RATE     = 2e-5  
EPOCHS            = 3   
WEIGHT_DECAY      = 0.02

# ── TASK PROMPT POOL FOR DE-COUPLING REALIGNMENT ────────────────────────
PROMPT_POOL = [
    "Trace the line of sight vector from the eyes of the subject. Provide the coordinate (X, Y) where their gaze lands.",
    "Look at the target person in the image frame. Predict their normalized gaze target coordinate as (X, Y) between 0.0 and 1.0.",
    "Determine the focus direction of the primary subject. Output the coordinate (X, Y) where they are looking."
]

# ── BOUNDED MULTIMODAL PROJECTION ADAPTER LAYER ─────────────────────────
class GazeMultimodalProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)  
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.projector:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print("[+] Projector weights initialized safely using bounded distributions.")
        
    def forward(self, x):
        return self.projector(x)

# ── REGRESSION HEAD: replaces text-token generation with direct coordinate
# output, on the hypothesis (see diagnostic evidence) that next-token
# cross-entropy over decimal-string tokens is a poor training signal for a
# continuous spatial task. Outputs: (x, y) via Sigmoid-bounded regression,
# PLUS a separate off-screen logit -- kept separate from (x,y) rather than
# encoded as (-1,-1), since cramming a sentinel into a [0,1]-bounded
# regression target would distort the loss landscape for genuine on-screen
# samples. Uses SmoothL1 for coords, matching the loss already used by the
# actual Stage-1 student, for methodological consistency.
class GazeCoordRegressionHead(nn.Module):
    def __init__(self, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.GELU(),
        )
        self.coord_out = nn.Linear(256, 2)      # (x, y), passed through sigmoid
        self.offscreen_out = nn.Linear(256, 1)  # off-screen logit (BCEWithLogits)
        self._init_weights()

    def _init_weights(self):
        for m in [self.net[0], self.coord_out, self.offscreen_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, pooled_hidden):
        feat = self.net(pooled_hidden)
        coords = torch.sigmoid(self.coord_out(feat))
        offscreen_logit = self.offscreen_out(feat)
        return coords, offscreen_logit

# ── PYTORCH MANIFEST DATASET ENGINE ─────────────────────────────────────
class GazeHybridDataset(Dataset):
    def __init__(self, dataframe, base_path):
        self.df = dataframe.reset_index(drop=True)
        self.base_path = base_path
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["img_path"])
        
        # Cross-platform environment path mapping check
        if not os.path.exists(img_path):
            show = str(row["show"])
            clip = str(row["clip"])
            fname = str(row["fname"])
            
            # Alternative relative path check for cluster structures
            for candidate in [
                os.path.join(self.base_path, "images", show, clip, fname),
                os.path.join(self.base_path, "images", clip, fname),
                os.path.join("videoattentiontarget", "images", show, clip, fname),
                os.path.join("images", show, clip, fname)
            ]:
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[-] Definitively missing source frame for manifest index {idx}: {img_path}")

        img_raw = Image.open(img_path).convert("RGB")
        w, h = img_raw.size
        
        gaze_x = float(row["gaze_x"])
        gaze_y = float(row["gaze_y"])

        if gaze_x != -1 and gaze_y != -1:
            norm_x, norm_y = gaze_x / w, gaze_y / h
            is_offscreen = 0.0
        else:
            # Off-screen: coord values are placeholders (masked out of the
            # coordinate loss via is_offscreen, not learned as literal -1s).
            norm_x, norm_y = 0.5, 0.5
            is_offscreen = 1.0

        prompt_text = random.choice(PROMPT_POOL)
        return {
            "image": img_raw,
            "prompt": prompt_text,
            "target_xy": torch.tensor([norm_x, norm_y], dtype=torch.float32),
            "is_offscreen": torch.tensor([is_offscreen], dtype=torch.float32),
        }

# ── GLOBAL COLLATION ENGINE FOR WINDOWS MULTIPROCESSING PICKLE COMPATIBILITY ──
def gaze_hybrid_collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "target_xy": torch.stack([item["target_xy"] for item in batch]),
        "is_offscreen": torch.stack([item["is_offscreen"] for item in batch]),
    }

# ── TRAIN EXECUTIVE ARCHITECTURE ────────────────────────────────────────
def main():
    global MANIFEST_PATH, BASE_PATH
    print("[*] Resolving Dataset Pipelines...")
    
    # Environment directory auto-discovery (Checks for Eureka2 scratch path targets)
    hpc_manifest = "/parallel_scratch/sl02092/standard_project/frame_manifest.csv"
    hpc_base = "/parallel_scratch/sl02092/standard_project/data/videoattentiontarget"
    
    if os.path.exists(hpc_manifest):
        MANIFEST_PATH = hpc_manifest
        print("[+] Detected Eureka2 Cluster Environment Manifest Layout.")
    else:
        for path_candidate in [MANIFEST_PATH, "frame_manifest.csv", "../frame_manifest.csv"]:
            if os.path.exists(path_candidate):
                MANIFEST_PATH = path_candidate
                break
                
    if os.path.exists(hpc_base):
        BASE_PATH = hpc_base
    elif not os.path.exists(BASE_PATH):
        if os.path.exists("videoattentiontarget"):
            BASE_PATH = "videoattentiontarget"
        
    full_df = pd.read_csv(MANIFEST_PATH)
    print(f"[+] Successfully loaded manifest layout ({len(full_df)} total available pairs).")
    
    if NUM_TRAIN_SAMPLES > 0 and len(full_df) > NUM_TRAIN_SAMPLES:
        train_df = full_df.sample(n=NUM_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"[+] Footprint configuration bound: Selected {NUM_TRAIN_SAMPLES} training instances.")
    else:
        train_df = full_df
        print(f"[*] Deep Processing Active: Consuming the entire dataset context ({len(train_df)} items).")

    print("[*] Initializing Foundation Weights & Tokenizers...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    full_vlm = AutoModelForImageTextToText.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    llm_model = full_vlm.model.language_model
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)

    dino_model.eval().requires_grad_(False)
    llm_model.eval().requires_grad_(False)
    
    dino_dim = dino_model.config.hidden_size
    llm_dim = llm_model.config.hidden_size
    
    proj_head = GazeMultimodalProjector(dino_dim, llm_dim).to("cuda:0", dtype=torch.bfloat16)
    proj_head.train().requires_grad_(True)
    
    reg_head = GazeCoordRegressionHead(llm_dim).to("cuda:0", dtype=torch.bfloat16)
    reg_head.train().requires_grad_(True)

    optimizer = torch.optim.AdamW(
        list(proj_head.parameters()) + list(reg_head.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    coord_criterion = nn.SmoothL1Loss()
    offscreen_criterion = nn.BCEWithLogitsLoss()
    
    dataset = GazeHybridDataset(train_df, BASE_PATH)
    
    # Handing over the top-level global function reference here:
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=gaze_hybrid_collate_fn, 
        num_workers=2, 
        drop_last=True
    )
    
    print(f"\n[+] System Grounded. Beginning Projector Alignment over {len(dataloader)} micro-batches...")
    
    start_tokens = llm_tokenizer("Image Context: [START_VISION] ", return_tensors="pt").to(llm_model.device)
    end_tokens   = llm_tokenizer(" [END_VISION]\nOperational Prompt: ", return_tensors="pt").to(llm_model.device)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            inputs_dino = dino_processor(images=batch["images"], return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                dino_out = dino_model(**inputs_dino)
                vis_features = dino_out.last_hidden_state[:, 1:, :]
                
            projected_vis = proj_head(vis_features.to(torch.bfloat16))

            target_xy = batch["target_xy"].to("cuda:0", dtype=torch.bfloat16)
            is_offscreen = batch["is_offscreen"].to("cuda:0", dtype=torch.bfloat16)

            batch_loss = 0.0
            for i in range(BATCH_SIZE):
                p_tokens = llm_tokenizer(batch["prompts"][i], return_tensors="pt").to(llm_model.device)

                embed_layer = llm_model.get_input_embeddings() if hasattr(llm_model, "get_input_embeddings") else llm_model.model.embed_tokens

                start_embeds = embed_layer(start_tokens.input_ids)
                end_embeds   = embed_layer(end_tokens.input_ids)
                p_embeds     = embed_layer(p_tokens.input_ids)

                # No target-text embeds: the model now sees image + prompt only,
                # and is read out via the pooled hidden state, not generation.
                combined_embeds = torch.cat([
                    start_embeds,
                    projected_vis[i:i+1],
                    end_embeds,
                    p_embeds,
                ], dim=1)

                outputs = llm_model(inputs_embeds=combined_embeds, attention_mask=None)
                hidden_states = outputs.last_hidden_state

                # Pool the LAST token position only -- this is where the LLM
                # has attended over the full image+prompt context already.
                pooled = hidden_states[:, -1, :]

                pred_xy, pred_offscreen_logit = reg_head(pooled)

                coord_loss = coord_criterion(pred_xy, target_xy[i:i+1])
                offscreen_loss = offscreen_criterion(pred_offscreen_logit, is_offscreen[i:i+1])

                # Mask coord loss out for off-screen samples (their xy target
                # is a placeholder, not a real location -- see dataset class).
                sample_loss = offscreen_loss + coord_loss * (1.0 - is_offscreen[i, 0])
                batch_loss += sample_loss
                
            batch_loss = batch_loss / BATCH_SIZE
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] | Step [{batch_idx+1}/{len(dataloader)}] -> Current Running Loss: {batch_loss.item():.4f}")

            # CHECKPOINT: save every 200 steps so a crash/timeout/preemption on the
            # HPC doesn't lose all progress. Saved under a distinct filename from
            # the final output, so a partial run is never mistaken for a complete one.
            if (batch_idx + 1) % 200 == 0:
                ckpt_path = os.path.join(
                    os.path.dirname(MANIFEST_PATH),
                    f"dino_internvl_REGRESSION_ckpt_epoch{epoch+1}_step{batch_idx+1}.pt"
                )
                torch.save({"proj_head": proj_head.state_dict(), "reg_head": reg_head.state_dict()}, ckpt_path)
                print(f"  [ckpt] Saved intermediate checkpoint -> {ckpt_path}")
            
        print(f"[*] Epoch [{epoch+1}/{EPOCHS}] Completed -> Average Realignment Loss: {total_loss / len(dataloader):.4f}")

        epoch_ckpt_path = os.path.join(os.path.dirname(MANIFEST_PATH), f"dino_internvl_REGRESSION_ckpt_epoch{epoch+1}_final.pt")
        torch.save({"proj_head": proj_head.state_dict(), "reg_head": reg_head.state_dict()}, epoch_ckpt_path)
        print(f"  [ckpt] Saved end-of-epoch checkpoint -> {epoch_ckpt_path}")
        
    # NOTE: deliberately a DIFFERENT filename from the text-generation version
    # (dino_internvl_projector.pt) -- this checkpoint has a different proj_head
    # state (trained against a different downstream head) and ALSO includes
    # reg_head, which the old eval/diagnostic script doesn't know how to load.
    # Loading this file with the OLD diag script would silently load a proj_head
    # that was never trained for text-generation use and produce garbage.
    output_weight_path = os.path.join(os.path.dirname(MANIFEST_PATH), "dino_internvl_REGRESSION_final.pt")
    torch.save({"proj_head": proj_head.state_dict(), "reg_head": reg_head.state_dict()}, output_weight_path)
    print(f"\n[+] Realignment complete. Model weight matrix exported to: {output_weight_path}")

if __name__ == "__main__":
    main()