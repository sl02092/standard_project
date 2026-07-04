"""
train_step43d7_hybrid.py — Alignment Optimization Harness
Trains a 2-layer MLP projection layer to align DINOv2 dense spatial tokens 
with InternVL3-8B-hf's text embedding space using Gaze target data.

Both massive backbones are completely frozen (Requires ~14GB VRAM for training 
due to activation pooling, fits safely within standard modern desktop GPUs).
"""

import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# ── CONFIGURATION ───────────────────────────────────────────────────────
BASE_PATH       = r"C:\repo\standard_project\videoattentiontarget"
DINO_MODEL_NAME = "facebook/dinov2-large"
LLM_MODEL_NAME  = "OpenGVLab/InternVL3-8B-hf"
BATCH_SIZE      = 2
LEARNING_RATE   = 5e-4
EPOCHS          = 5

# Using a subset of your target sequences for explicit alignment training
TRAIN_FRAMES = [
    ("13525_13575", 13567, "s00"),
    ("13525_13575", 13570, "s00"),
    ("13525_13575", 13575, "s00"),
    ("2250_2300",  2295, "s00"),
    ("2250_2300",  2300, "s00"),
    ("1650_1775",  1673, "s00"),
    ("1650_1775",  1700, "s00"),
    ("1650_1775",  1728, "s00"),
]

# ── MULTIMODAL PROJECTION ADAPTER LAYER ─────────────────────────────────
class GazeMultimodalProjector(nn.Module):
    """
    Two-layer MLP network with non-linear activation. Maps raw physical 
    vision vectors smoothly into textual token embedding space manifolds.
    """
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, x):
        return self.projector(x)

# ── PYTORCH REUSABLE DATASET ENGINE ─────────────────────────────────────
class GazeHybridDataset(Dataset):
    def __init__(self, frames_list, base_path):
        self.frames = frames_list
        self.base_path = base_path
        
    def _find_dir(self, root, clip_id):
        for r, dirs, _ in os.walk(root):
            for d in dirs:
                if d.strip() == str(clip_id).strip():
                    return os.path.join(r, d)
        return None

    def _parse_txt(self, filepath, frame_num):
        if not os.path.exists(filepath): return None
        with open(filepath, "r") as f:
            for line in f:
                tokens = line.strip().split(',')
                if len(tokens) < 7: continue
                try:
                    if int(re.sub(r"\D", "", os.path.basename(tokens[0]))) == int(frame_num):
                        return {
                            "fname": os.path.basename(tokens[0]),
                            "gaze_x": float(tokens[5]), "gaze_y": float(tokens[6])
                        }
                except ValueError: continue
        return None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        clip, frame, subject = self.frames[idx]
        ann_dir = self._find_dir(os.path.join(self.base_path, "annotations"), clip)
        img_dir = self._find_dir(os.path.join(self.base_path, "images"), clip)
        
        data = self._parse_txt(os.path.join(ann_dir, f"{subject}.txt"), frame)
        img_path = os.path.join(img_dir, data["fname"])
        img_raw = Image.open(img_path).convert("RGB")
        w, h = img_raw.size
        
        # Format exact target text response string for alignment optimization
        if data["gaze_x"] != -1 and data["gaze_y"] != -1:
            target_text = f"({data['gaze_x']/w:.3f}, {data['gaze_y']/h:.3f})"
        else:
            target_text = "off-screen"
            
        prompt_text = "Trace the line of sight vector from the eyes of the subject. Provide the coordinate (X, Y) where their gaze lands."
        
        return {"image": img_raw, "prompt": prompt_text, "target": target_text}

# ── TRAIN EXECUTIVE ARCHITECTURE ────────────────────────────────────────
def main():
    print("[*] Initializing Pipeline Engines...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    print(f"[*] Loading VLM to extract language decoder: {LLM_MODEL_NAME}...")
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
    
    dataset = GazeHybridDataset(TRAIN_FRAMES, BASE_PATH)
    
    # Custom collate function to handle PIL images natively inside a batch dictionary
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
                
                # Fetch text tokens embeddings from base lookup matrices
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
                # 1. Run the base language model backbone to get hidden states
                outputs = llm_model(inputs_embeds=combined_embeds)
                
                # 2. Extract final hidden states and pass them through the VLM's language head
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