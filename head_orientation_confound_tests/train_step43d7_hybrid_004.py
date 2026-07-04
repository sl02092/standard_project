"""
train_step43d7_hybrid_004.py — Cross-Platform Production Optimization Harness
Trains a 2-layer MLP projection layer to align DINOv2 dense spatial tokens 
with InternVL3-8B-hf's text embedding space using Gaze target data.

FIXES:
  - Global Collate Scope: Fixed Windows Popen/Spawn AttributeError by lifting 
    the DataLoader collation engine out of main() into the module global scope.
  - Multi-Environment Dataset Resolution: Fallbacks safely between Windows absolute 
    paths and standardized Linux/HPC relative data trees.
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
NUM_TRAIN_SAMPLES = -1  
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
            target_text = f"({gaze_x/w:.3f}, {gaze_y/h:.3f})"
        else:
            target_text = "off-screen"
            
        prompt_text = random.choice(PROMPT_POOL)
        return {"image": img_raw, "prompt": prompt_text, "target": target_text}

# ── GLOBAL COLLATION ENGINE FOR WINDOWS MULTIPROCESSING PICKLE COMPATIBILITY ──
def gaze_hybrid_collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "targets": [item["target"] for item in batch]
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
    
    optimizer = torch.optim.AdamW(proj_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
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
            
            batch_loss = 0.0
            for i in range(BATCH_SIZE):
                p_tokens = llm_tokenizer(batch["prompts"][i], return_tensors="pt").to(llm_model.device)
                t_tokens = llm_tokenizer(batch["targets"][i], return_tensors="pt").to(llm_model.device)
                
                embed_layer = llm_model.get_input_embeddings() if hasattr(llm_model, "get_input_embeddings") else llm_model.model.embed_tokens
                
                start_embeds = embed_layer(start_tokens.input_ids)
                end_embeds   = embed_layer(end_tokens.input_ids)
                p_embeds     = embed_layer(p_tokens.input_ids)
                t_embeds     = embed_layer(t_tokens.input_ids)
                
                combined_embeds = torch.cat([
                    start_embeds,          
                    projected_vis[i:i+1],  
                    end_embeds,            
                    p_embeds,              
                    t_embeds               
                ], dim=1)
                
                start_len = start_tokens.input_ids.size(1)
                vis_len   = projected_vis.size(1)
                end_len   = end_tokens.input_ids.size(1)
                p_len     = p_tokens.input_ids.size(1)
                t_len     = t_tokens.input_ids.size(1)
                
                total_len = start_len + vis_len + end_len + p_len + t_len
                
                labels = torch.full((1, total_len), -100, dtype=torch.long, device=llm_model.device)
                labels[0, start_len + vis_len + end_len + p_len:] = t_tokens.input_ids[0]
                
                outputs = llm_model(inputs_embeds=combined_embeds, attention_mask=None)
                hidden_states = outputs.last_hidden_state
                logits = full_vlm.lm_head(hidden_states)                
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                batch_loss += loss
                
            batch_loss = batch_loss / BATCH_SIZE
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] | Step [{batch_idx+1}/{len(dataloader)}] -> Current Running Loss: {batch_loss.item():.4f}")
            
        print(f"[*] Epoch [{epoch+1}/{EPOCHS}] Completed -> Average Realignment Loss: {total_loss / len(dataloader):.4f}")
        
    output_weight_path = os.path.join(os.path.dirname(MANIFEST_PATH), "dino_internvl_projector.pt")
    torch.save(proj_head.state_dict(), output_weight_path)
    print(f"\n[+] Realignment complete. Model weight matrix exported to: {output_weight_path}")

if __name__ == "__main__":
    main()