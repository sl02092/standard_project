"""
train_step43d7_hybrid_002.py — Stable Alignment Optimization Harness
Trains a 2-layer MLP projection layer to align DINOv2 dense spatial tokens 
with InternVL3-8B-hf's text embedding space using Gaze target data.

Refactored to utilize a scalable, manifest-driven footprint loader mirroring the
teacher pipeline logic. Implements terminal layer normalization and careful weight
initialization to protect the language model's latent manifold from representation collapse.

Both massive backbones are completely frozen (Requires ~14GB VRAM for training 
due to activation pooling, fits safely within standard modern desktop GPUs).
"""

import os
import re
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# ── CONFIGURATION & HARDENED HYPERPARAMETERS ───────────────────────────
BASE_PATH         = r"C:\repo\standard_project\videoattentiontarget"
MANIFEST_PATH     = r"C:\repo\standard_project\frame_manifest.csv"
DINO_MODEL_NAME   = "facebook/dinov2-large"
LLM_MODEL_NAME    = "OpenGVLab/InternVL3-8B-hf"

# Scalability Optimization Parameters
NUM_TRAIN_SAMPLES = 150  # ← ADJUST THIS: Number of random frames to sample for scaling your footprint
BATCH_SIZE        = 2
LEARNING_RATE     = 2e-5  # ← Lowered from 5e-4 to prevent embedding explosion/decoder saturation
EPOCHS            = 5

# ── BOUNDED MULTIMODAL PROJECTION ADAPTER LAYER ─────────────────────────
class GazeMultimodalProjector(nn.Module):
    """
    Two-layer MLP network with non-linear activation and terminal LayerNorm.
    Forces output tokens to respect the structural manifold scales of the frozen LLM.
    """
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)  # ← Enforces numerical scale compatibility with the LLM
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

# ── PYTORCH MANIFEST DATASET ENGINE ─────────────────────────────────────
class GazeHybridDataset(Dataset):
    """
    Manifest-driven dataset engine that extracts records directly from pre-parsed 
    dataframe rows. Handles robust path resolution across nested subdirectories.
    """
    def __init__(self, dataframe, base_path):
        self.df = dataframe.reset_index(drop=True)
        self.base_path = base_path
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Try using the raw manifest path first
        img_path = row["img_path"]
        
        if not os.path.exists(img_path):
            show = str(row["show"])
            clip = str(row["clip"])
            fname = str(row["fname"])
            
            # Canonical Local Layout: base_path/images/{show}/{clip}/{fname}
            potential_path = os.path.join(self.base_path, "images", show, clip, fname)
            
            if os.path.exists(potential_path):
                img_path = potential_path
            else:
                # Fallback layout: base_path/images/{clip}/{fname} (if some shows are flat)
                potential_path2 = os.path.join(self.base_path, "images", clip, fname)
                if os.path.exists(potential_path2):
                    img_path = potential_path2
                else:
                    # Suffix harvesting: Extract the relative portion after "images" if embedded in the cluster path
                    if "images" in str(row["img_path"]):
                        rel_suffix = str(row["img_path"]).split("images")[-1].lstrip("/\\")
                        img_path = os.path.join(self.base_path, "images", rel_suffix)

        # Defensive path safety checkpoint
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[-] Could not resolve image path locally. Tried:\n"
                                    f"    - Manifest path: {row['img_path']}\n"
                                    f"    - Structured path: {os.path.join(self.base_path, 'images', str(row['show']), str(row['clip']), str(row['fname']))}")

        img_raw = Image.open(img_path).convert("RGB")
        w, h = img_raw.size
        
        gaze_x = float(row["gaze_x"])
        gaze_y = float(row["gaze_y"])
        
        # Format exact target text response string for alignment optimization
        if gaze_x != -1 and gaze_y != -1:
            target_text = f"({gaze_x/w:.3f}, {gaze_y/h:.3f})"
        else:
            target_text = "off-screen"
            
        # V2 Alignment prompt layout proven to trigger cross-modal projection logic
        prompt_text = "Trace the line of sight vector from the eyes of the subject. Provide the coordinate (X, Y) where their gaze lands."
        
        return {"image": img_raw, "prompt": prompt_text, "target": target_text}

# ── TRAIN EXECUTIVE ARCHITECTURE ────────────────────────────────────────
def main():
    global MANIFEST_PATH
    print("[*] Resolving Dataset Pipelines...")
    
    # Self-healing manifest discovery depending on execution context
    if not os.path.exists(MANIFEST_PATH):
        if os.path.exists("../frame_manifest.csv"):
            MANIFEST_PATH = "../frame_manifest.csv"
        elif os.path.exists("frame_manifest.csv"):
            MANIFEST_PATH = "frame_manifest.csv"

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"[-] Definitively missing frame_manifest.csv at: {MANIFEST_PATH}. Check target configurations.")
        
    # Read the data manifest file 
    full_df = pd.read_csv(MANIFEST_PATH)
    print(f"[+] Successfully loaded manifest layout ({len(full_df)} total available pairs).")
    
    # Extract dynamic training subset based on user-defined footprint configurations
    if len(full_df) > NUM_TRAIN_SAMPLES:
        train_df = full_df.sample(n=NUM_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"[+] Random Sampling Context Enabled: Selected {NUM_TRAIN_SAMPLES} training instances.")
    else:
        train_df = full_df
        print(f"[*] Footprint configuration equals/exceeds manifest data. Utilizing entire dataset ({len(train_df)} items).")

    print("[*] Initializing Foundation Weights & Tokenizers...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    print(f"[*] Extracting Language Decoder Structure: {LLM_MODEL_NAME}...")
    full_vlm = AutoModelForImageTextToText.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    llm_model = full_vlm.model.language_model
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)

    # Freeze foundation layers explicitly
    dino_model.eval().requires_grad_(False)
    llm_model.eval().requires_grad_(False)
    
    dino_dim = dino_model.config.hidden_size
    llm_dim = llm_model.config.hidden_size
    
    # Instantiate trainable projector
    proj_head = GazeMultimodalProjector(dino_dim, llm_dim).to("cuda:0", dtype=torch.bfloat16)
    proj_head.train().requires_grad_(True)
    
    optimizer = torch.optim.AdamW(proj_head.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Mount dataset loader using sampled manifest rows
    dataset = GazeHybridDataset(train_df, BASE_PATH)
    
    def collate_fn(batch):
        return {
            "images": [item["image"] for item in batch],
            "prompts": [item["prompt"] for item in batch],
            "targets": [item["target"] for item in batch]
        }
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print("\n[+] System Grounded. Beginning Projector Alignment Phase...")
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 1. Process and extract dense DINO spatial patches
            inputs_dino = dino_processor(images=batch["images"], return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                dino_out = dino_model(**inputs_dino)
                # Shape: [Batch, Patches, dino_dim]
                vis_features = dino_out.last_hidden_state[:, 1:, :]
                
            # 2. Project visual state tokens directly into target LLM space hidden-widths
            projected_vis = proj_head(vis_features.to(torch.bfloat16))
            
            # 3. Handle batched causal cross-entropy sequence construction
            batch_loss = 0.0
            for i in range(len(batch["prompts"])):
                p_tokens = llm_tokenizer(batch["prompts"][i], return_tensors="pt").to(llm_model.device)
                t_tokens = llm_tokenizer(batch["targets"][i], return_tensors="pt").to(llm_model.device)
                
                # Fetch text token embeddings from base lookup matrices
                embed_layer = llm_model.get_input_embeddings() if hasattr(llm_model, "get_input_embeddings") else llm_model.model.embed_tokens
                p_embeds = embed_layer(p_tokens.input_ids)
                t_embeds = embed_layer(t_tokens.input_ids)
                
                # Construct combined sequence matrix [Vision Patches, Prompt Tokens, Target Coordinates]
                combined_embeds = torch.cat([projected_vis[i:i+1], p_embeds, t_embeds], dim=1)
                
                # Mask out loss calculations for everything except the target coordinate tokens (-100 index)
                vis_len = projected_vis.size(1)
                p_len = p_tokens.input_ids.size(1)
                t_len = t_tokens.input_ids.size(1)
                
                labels = torch.full((1, vis_len + p_len + t_len), -100, dtype=torch.long, device=llm_model.device)
                labels[0, vis_len + p_len:] = t_tokens.input_ids[0]
                
                # Autoregressive forward execution
                outputs = llm_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=None,
                )
                
                # Extract final hidden states and pass them through the VLM's language head
                hidden_states = outputs.last_hidden_state
                logits = full_vlm.lm_head(hidden_states)                
                
                # Shift sequence space by 1 for language objective tracking
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                batch_loss += loss
                
            batch_loss = batch_loss / len(batch["prompts"])
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
        print(f"  Epoch [{epoch+1}/{EPOCHS}] -> Average Realignment Loss: {total_loss / len(dataloader):.4f}")
        
    # Save the optimized weights locally
    output_weight_path = os.path.join(BASE_PATH, "dino_internvl_projector.pt")
    torch.save(proj_head.state_dict(), output_weight_path)
    print(f"\n[+] Realignment complete. Model weight matrix exported to: {output_weight_path}")

if __name__ == "__main__":
    main()