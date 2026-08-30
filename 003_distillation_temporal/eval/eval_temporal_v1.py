"""
eval_temporal_v1.py — scores an already-trained temporal head checkpoint
(from train_temporal_v1.py's new model_{architecture}.pt output) against
UCO-LAEO, with NO training. The temporal equivalent of eval_static_v1.py.

REQUIRES train_temporal_v1.py's checkpoint-saving fix (added 2026-08-10)
-- checkpoints trained before that fix don't exist and can't be loaded
here; they'd need retraining first.

MODEL LAYER: GRUHead/TransformerHead/ResidualGRUHead/ResidualTransformerHead
and their helper functions are copied verbatim from train_temporal_v1.py
(same source, same date) rather than imported -- train_temporal_v1.py has
its own hard module-level config requirements (TEMPORAL_DATASETS etc.)
that would fire on import, the same fragile-import problem already found
and fixed once for distillation_v1.py's GazeStudent/GazeDataset. If you
change one copy, change both.

REQUIRES UCO-LAEO's cached embeddings to already exist (run
extract_temporal_features_v1.py with EXTRACT_DATASET=ucolaeo first, once
per static condition being tested) -- this script only reads that cache,
it doesn't create it.

Usage: set env vars below, then run once per (static_condition,
architecture, horizon) combination you want scored -- or loop externally
(see the walkthrough for a script that loops all of them).
"""

import os
import json
import math

import numpy as np
import torch
import torch.nn as nn

WINDOW_LENGTH = 8

# Must match extract_temporal_features_v1.py's own PATH_REMAP exactly --
# see that script's comment for the full story.
PATH_REMAP = {
    "ucolaeo": (
        r"C:\repo\UCO-LAEO\ucolaeodb\frames",
        "/parallel_scratch/sl02092/standard_project/data/ucolaeodb/frames",
    ),
}

# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

FEATURE_CACHE_ROOT = os.environ.get("FEATURE_CACHE_ROOT")
if not FEATURE_CACHE_ROOT:
    raise ValueError("FEATURE_CACHE_ROOT not set.")

STATIC_CONDITION = os.environ.get("STATIC_CONDITION")
if not STATIC_CONDITION:
    raise ValueError("STATIC_CONDITION not set.")

TEMPORAL_CHECKPOINT_DIR = os.environ.get("TEMPORAL_CHECKPOINT_DIR")
if not TEMPORAL_CHECKPOINT_DIR:
    raise ValueError("TEMPORAL_CHECKPOINT_DIR not set -- e.g. "
                      "temporal_output_vittiny/gt/horizon_300ms (must match "
                      "the horizon of the checkpoint being loaded).")

TEMPORAL_ARCHITECTURES = [a.strip() for a in os.environ.get(
    "TEMPORAL_ARCHITECTURES", "GRU,Transformer,ResidualGRU,ResidualTransformer").split(",")]

TARGET_HORIZON_MS = int(os.environ.get("TARGET_HORIZON_MS", "300"))
UCOLAEO_TEMPORAL_MANIFEST_PATH = os.environ.get("UCOLAEO_TEMPORAL_MANIFEST_PATH")
if not UCOLAEO_TEMPORAL_MANIFEST_PATH:
    raise ValueError("UCOLAEO_TEMPORAL_MANIFEST_PATH not set.")

OUTPUT_DIR = os.environ.get("EVAL_TEMPORAL_OUTPUT_DIR")
if not OUTPUT_DIR:
    raise ValueError("EVAL_TEMPORAL_OUTPUT_DIR not set.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ══════════════════════════════════════════════════════════════════════
# ── MODEL LAYER — copied verbatim from train_temporal_v1.py (see docstring) ──
# ══════════════════════════════════════════════════════════════════════

def make_regression_tail(input_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2), nn.Sigmoid(),
    )


def make_residual_tail(input_dim, dropout=0.1):
    tail = nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2),
    )
    nn.init.zeros_(tail[-1].weight)
    nn.init.zeros_(tail[-1].bias)
    return tail


def pick_nhead(input_dim, max_heads=8):
    for h in range(min(max_heads, input_dim), 0, -1):
        if input_dim % h == 0:
            return h
    return 1


class GRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.tail = make_regression_tail(hidden_dim, dropout)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.tail(h_n[-1])


class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256,
                 dropout=0.1, window_length=WINDOW_LENGTH):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_regression_tail(input_dim, dropout)

    def forward(self, x):
        x = x + self.pos_embed
        out = self.encoder(x)
        return self.tail(out[:, -1, :])


class ResidualGRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.tail = make_residual_tail(hidden_dim, dropout)

    def forward(self, x):
        reference = x[:, -1, -2:]
        _, h_n = self.gru(x)
        delta = self.tail(h_n[-1])
        return torch.clamp(reference + delta, 0.0, 1.0)


class ResidualTransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256,
                 dropout=0.1, window_length=WINDOW_LENGTH):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_residual_tail(input_dim, dropout)

    def forward(self, x):
        reference = x[:, -1, -2:]
        x2 = x + self.pos_embed
        out = self.encoder(x2)
        delta = self.tail(out[:, -1, :])
        return torch.clamp(reference + delta, 0.0, 1.0)


MODEL_CLASSES = {
    "GRU": GRUHead, "Transformer": TransformerHead,
    "ResidualGRU": ResidualGRUHead, "ResidualTransformer": ResidualTransformerHead,
}


def load_temporal_model(architecture, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        return None, None
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    input_dim = ckpt["input_dim"]
    cls = MODEL_CLASSES[architecture]
    if architecture in ("Transformer", "ResidualTransformer"):
        model = cls(input_dim=input_dim, nhead=pick_nhead(input_dim)).to(DEVICE)
    else:
        model = cls(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[load] {architecture}: trained_horizon={ckpt.get('target_horizon_ms')}ms "
          f"trained_condition={ckpt.get('static_condition')} "
          f"trained_on={ckpt.get('datasets')} best_ade={ckpt.get('best_ade'):.4f}")
    return model, ckpt


# ══════════════════════════════════════════════════════════════════════
# ── UCO-LAEO DATA — same horizon filtering as JointTemporalDataset, one dataset ──
# ══════════════════════════════════════════════════════════════════════

def build_cache_key(clip_id, subject, fname):
    return f"{clip_id}/{subject}/{fname}"  # UCO-LAEO's own key format -- no show/clip split


def load_ucolaeo_windows(condition, target_horizon_ms):
    cache_dir = os.path.join(FEATURE_CACHE_ROOT, "ucolaeo", condition)
    index_path = os.path.join(cache_dir, "frame_features_index.jsonl")
    embeddings = np.load(os.path.join(cache_dir, "frame_features.npy"))
    predicted_xy = np.load(os.path.join(cache_dir, "frame_predicted_xy.npy"))

    key_to_idx = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key_to_idx[r["key"]] = r["embedding_idx"]

    rows = []
    counts = {"total": 0, "wrong_horizon": 0, "invalid_window": 0,
              "offscreen_target": 0, "missing_embedding": 0, "kept": 0}
    with open(UCOLAEO_TEMPORAL_MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts["total"] += 1
            if row.get("target_horizon_ms") != target_horizon_ms:
                counts["wrong_horizon"] += 1
                continue
            if not row["window_valid"]:
                counts["invalid_window"] += 1
                continue
            if row["target_gaze_x"] is None:
                counts["offscreen_target"] += 1
                continue

            clip_id, subject = row["clip_id"], row["subject_id"]
            fnames = [os.path.basename(p) for p in row["context_frame_paths"]]
            keys = [build_cache_key(clip_id, subject, fn) for fn in fnames]
            if not all(k in key_to_idx for k in keys):
                counts["missing_embedding"] += 1
                continue

            embed_indices = [key_to_idx[k] for k in keys]
            rows.append({
                "clip": clip_id, "subject": subject,
                "embed_seq": np.concatenate(
                    [embeddings[embed_indices], predicted_xy[embed_indices]], axis=1),
                "target_gaze_x": float(row["target_gaze_x"]),
                "target_gaze_y": float(row["target_gaze_y"]),
                "target_gaze_type": row["target_gaze_type"],
                "last_context_raw_path": row["context_frame_paths"][-1],
            })
            counts["kept"] += 1

    print(f"[ucolaeo] {counts['kept']} usable rows (target_horizon_ms={target_horizon_ms}). "
          f"Dropped: {counts['wrong_horizon']} wrong-horizon, {counts['invalid_window']} "
          f"invalid-window, {counts['offscreen_target']} offscreen-target, "
          f"{counts['missing_embedding']} missing-embedding.")
    return rows


def resolve_dims(rows):
    """Real per-clip image dimensions for target normalization -- same
    need as train_temporal_v1.py's own JointTemporalDataset. FOUND
    MISSING 2026-08-12: UCO-LAEO's manifest has the same baked-in local
    Windows path problem as ChildPlay/VACATION (see
    extract_temporal_features_v1.py's own PATH_REMAP for the full story)
    -- this function was opening the raw, unremapped manifest path
    directly, so every lookup failed and every row got silently dropped.
    Same PATH_REMAP config as the other two scripts, applied here too."""
    from PIL import Image

    local_prefix, hpc_prefix = PATH_REMAP.get("ucolaeo", (None, None))

    def remap(raw_path):
        if local_prefix and hpc_prefix:
            normalized = raw_path.replace("\\", "/")
            normalized_prefix = local_prefix.replace("\\", "/")
            if normalized.startswith(normalized_prefix):
                return hpc_prefix.rstrip("/") + "/" + normalized[len(normalized_prefix):].lstrip("/")
        return raw_path

    dims = {}
    for r in rows:
        clip = r["clip"]
        if clip in dims:
            continue
        resolved_path = remap(r["last_context_raw_path"])
        try:
            with Image.open(resolved_path) as img:
                dims[clip] = img.size
        except Exception as e:
            print(f"[ucolaeo] WARNING: couldn't read dimensions for clip={clip}: {e}")
    return dims


# ══════════════════════════════════════════════════════════════════════
# ── EVAL ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model, rows, dims):
    dists, by_gaze_type = [], {}
    with torch.no_grad():
        for r in rows:
            w, h = dims.get(r["clip"], (None, None))
            if w is None:
                continue
            emb = torch.from_numpy(r["embed_seq"]).float().unsqueeze(0).to(DEVICE)
            target = torch.tensor([[r["target_gaze_x"] / w, r["target_gaze_y"] / h]]).to(DEVICE)
            pred = model(emb)
            d = torch.sqrt(((pred - target) ** 2).sum()).item()
            dists.append(d)
            by_gaze_type.setdefault(r["target_gaze_type"], []).append(d)
    return dists, by_gaze_type


def main():
    print(f"Static condition: {STATIC_CONDITION}  Horizon: {TARGET_HORIZON_MS}ms")
    print(f"Architectures: {TEMPORAL_ARCHITECTURES}\n")

    rows = load_ucolaeo_windows(STATIC_CONDITION, TARGET_HORIZON_MS)
    if len(rows) < 10:
        print("ERROR: too few usable UCO-LAEO rows -- check the cache/manifest before trusting this.")
        return
    dims = resolve_dims(rows)
    if len(dims) == 0:
        raise RuntimeError(
            "0 clips resolved real image dimensions -- every row would be "
            "dropped and every downstream ADE would be undefined. Almost "
            "certainly means PATH_REMAP['ucolaeo'] doesn't match this "
            "manifest's actual local path prefix -- check the WARNING "
            "lines above for the exact unresolved path and compare "
            "against PATH_REMAP's local_prefix."
        )

    all_results = {}
    for arch in TEMPORAL_ARCHITECTURES:
        ckpt_path = os.path.join(TEMPORAL_CHECKPOINT_DIR, f"model_{arch}.pt")
        model, ckpt = load_temporal_model(arch, ckpt_path)
        if model is None:
            print(f"  SKIPPED {arch} -- no checkpoint at {ckpt_path}")
            continue

        if ckpt.get("target_horizon_ms") != TARGET_HORIZON_MS:
            print(f"  WARNING: {arch} checkpoint was trained at "
                  f"{ckpt.get('target_horizon_ms')}ms, evaluating against "
                  f"{TARGET_HORIZON_MS}ms UCO-LAEO windows -- these should match; "
                  f"double check TEMPORAL_CHECKPOINT_DIR/TARGET_HORIZON_MS.")

        dists, by_gaze_type = evaluate_model(model, rows, dims)
        pooled_ade = sum(dists) / len(dists)
        by_gt_ade = {g: sum(v) / len(v) for g, v in by_gaze_type.items()}
        print(f"  {arch:<22} pooled_ADE={pooled_ade:.4f}  n={len(dists)}  "
              f"by_gaze_type={ {g: round(v, 4) for g, v in by_gt_ade.items()} }")
        all_results[arch] = {"pooled_ade": pooled_ade, "n": len(dists), "by_gaze_type": by_gt_ade}

    output = {
        "run_metadata": {"static_condition": STATIC_CONDITION, "target_horizon_ms": TARGET_HORIZON_MS,
                          "architectures": TEMPORAL_ARCHITECTURES, "n_rows": len(rows)},
        "results": all_results,
    }
    out_path = os.path.join(OUTPUT_DIR, "eval_temporal_ucolaeo.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
