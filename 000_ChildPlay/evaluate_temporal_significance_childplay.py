"""
evaluate_temporal_significance_childplay.py — answers two questions:
1. Is GRU's SANITY-margin over global-mean (LR=1e-5) real, or noise?
2. Does VAT's object-gaze-denoising / social-gaze-cost pattern replicate
   on ChildPlay, against static-copy-forward (the PRIMARY, fair baseline)?
Run from the same directory as the other ChildPlay scripts.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from temporal_window_dataset_childplay import TemporalWindowDataset, build_clip_split
from train_temporal_poc_childplay import (
    GRUHead, ResidualGRUHead, ResidualTransformerHead,
    train_one_model, evaluate, compute_baseline_computable_mask,
    seed_everything, pick_nhead, BATCH_SIZE, SEED,
)

N_RESAMPLES = 10000

def bootstrap_ci(diffs, n_resamples=N_RESAMPLES, seed=42):
    """diffs[i] = baseline_dist[i] - model_dist[i]; positive = model wins."""
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(seed)
    resampled = rng.choice(diffs, size=(n_resamples, len(diffs)), replace=True)
    means = resampled.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return diffs.mean(), lo, hi, (lo > 0 or hi < 0)

def main():
    train_ids, val_ids = build_clip_split()
    train_ds = TemporalWindowDataset(split_clip_ids=train_ids, split_name="train")
    val_ds = TemporalWindowDataset(split_clip_ids=val_ids, split_name="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = next(iter(train_loader))["embeddings"].shape[-1]
    matched_mask = compute_baseline_computable_mask(val_ds)
    matched_idx = [i for i, m in enumerate(matched_mask) if m]
    gaze_types = [val_ds[i]["target_gaze_type"] for i in range(len(val_ds))]
    print(f"{len(matched_idx)}/{len(matched_mask)} val rows matched.\n")

    train_targets = torch.stack([train_ds[i]["target"] for i in range(len(train_ds))])
    global_mean_xy = train_targets.mean(dim=0)
    val_targets = torch.stack([val_ds[i]["target"] for i in range(len(val_ds))])
    gm_dists = torch.sqrt(((val_targets - global_mean_xy) ** 2).sum(dim=1)).numpy()

    scf_dists = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        d = torch.sqrt(((item["embeddings"][-1, -2:] - item["target"]) ** 2).sum())
        scf_dists.append(d.item())
    scf_dists = np.array(scf_dists)

    results = {}
    for name, ctor in [
        ("GRU", lambda: GRUHead(input_dim=input_dim)),
        ("ResidualGRU", lambda: ResidualGRUHead(input_dim=input_dim)),
        ("ResidualTransformer", lambda: ResidualTransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))),
    ]:
        seed_everything(SEED)
        model, best_ade, best_epoch = train_one_model(name, model=ctor(), train_loader=train_loader, val_loader=val_loader)
        results[name] = np.array(evaluate(model, val_loader))
        print(f"[{name}] best_epoch={best_epoch+1}, matched_ADE={results[name][matched_idx].mean():.4f}")

    print("\n" + "=" * 70 + "\nQ1 -- GRU vs global-mean, SANITY margin\n" + "=" * 70)
    diffs = gm_dists[matched_idx] - results["GRU"][matched_idx]
    m, lo, hi, sig = bootstrap_ci(diffs)
    print(f"  mean diff={m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"{'SIGNIFICANT' if sig else 'NOT significant -- indistinguishable from noise'}")

    print("\n" + "=" * 70 + "\nQ2 -- gaze-type breakdown vs static-copy-forward\n" + "=" * 70)
    for gtype in ("object", "social"):
        idx = [i for i in matched_idx if gaze_types[i] == gtype]
        if not idx:
            continue
        scf_sub = scf_dists[idx].mean()
        print(f"\n  -- {gtype} (n={len(idx)}) -- static-copy-forward: {scf_sub:.4f}")
        for name in ("ResidualGRU", "ResidualTransformer"):
            model_sub = results[name][idx].mean()
            pct = (scf_sub - model_sub) / scf_sub * 100
            m, lo, hi, sig = bootstrap_ci(scf_dists[idx] - results[name][idx])
            print(f"  {name:22s}: {model_sub:.4f} ({pct:+.1f}%)  95% CI [{lo:.4f}, {hi:.4f}]  "
                  f"{'SIGNIFICANT' if sig else 'not significant'}")

if __name__ == "__main__":
    main()