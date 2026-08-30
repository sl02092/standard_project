"""
train_temporal_poc_childplay_balanced.py — class-balanced retrain, testing
whether the large object/social asymmetry in the LR=1e-5 gaze-type
breakdown is a training-imbalance artifact (74.8%/25.2% object/social)
rather than genuine denoising value.

Reuses every model class, baseline computation, and evaluation function
from train_temporal_poc_childplay.py UNCHANGED (imported, not redefined)
-- only the training loop changes, applying an inverse-frequency
per-sample loss weight keyed on each row's target_gaze_type. LR, epochs,
seeding discipline, and architectures are identical to the run already
done, so this is a clean, single-variable comparison against it.
"""

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from temporal_window_dataset_childplay import (
    TemporalWindowDataset, build_clip_split, MANIFEST_PATH,
)
from train_temporal_poc_childplay import (
    GRUHead, ResidualGRUHead, ResidualTransformerHead,
    count_params, pick_nhead, seed_everything, evaluate,
    compute_baseline_computable_mask,
    DEVICE, EPOCHS, BATCH_SIZE, LR, SEED,
)


def train_one_model_weighted(name, model, train_loader, val_loader, class_weights,
                              epochs=EPOCHS, lr=LR):
    """Identical to train_temporal_poc_childplay.py's train_one_model(),
    except the loss is computed per-sample (reduction='none') and
    multiplied by each sample's class weight before averaging."""
    model.to(DEVICE)
    print(f"\n[{name}] {count_params(model):,} trainable parameters (class-balanced loss).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    best_val_ade = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            emb = batch["embeddings"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            pred = model(emb)
            per_sample_loss = loss_fn(pred, target).mean(dim=1)  # (B,)
            weights = torch.tensor(
                [class_weights[t] for t in batch["target_gaze_type"]],
                device=DEVICE, dtype=torch.float32,
            )
            loss = (per_sample_loss * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_dists = evaluate(model, val_loader)
        val_ade = sum(val_dists) / len(val_dists)
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{name}] epoch {epoch+1:>3}/{epochs}  "
                  f"train_loss={sum(train_losses)/len(train_losses):.4f}  val_ADE={val_ade:.4f}")

    model.load_state_dict(best_state)
    print(f"[{name}] Best val_ADE={best_val_ade:.4f} at epoch {best_epoch+1}/{epochs}")
    return model, best_val_ade, best_epoch


def main():
    train_ids, val_ids = build_clip_split(MANIFEST_PATH)
    train_ds = TemporalWindowDataset(split_clip_ids=train_ids, split_name="train")
    val_ds = TemporalWindowDataset(split_clip_ids=val_ids, split_name="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    input_dim = next(iter(train_loader))["embeddings"].shape[-1]

    # Weights computed from THIS training set, not hardcoded from the
    # 74.8/25.2 figures already seen, in case it regenerates slightly differently.
    counts = Counter(r["target_gaze_type"] for r in train_ds.rows)
    total = sum(counts.values())
    class_weights = {g: total / (len(counts) * n) for g, n in counts.items()}
    print("Class-balanced loss weights (inverse frequency):")
    for g, n in counts.most_common():
        print(f"  {g:10s}: n={n:6d} ({n/total*100:.1f}%)  weight={class_weights[g]:.3f}")

    matched_mask = compute_baseline_computable_mask(val_ds)
    n_matched = sum(matched_mask)
    gaze_types = [val_ds[i]["target_gaze_type"] for i in range(len(val_ds))]
    print(f"\n{n_matched}/{len(matched_mask)} val rows matched.\n")

    train_targets = torch.stack([train_ds[i]["target"] for i in range(len(train_ds))])
    global_mean_xy = train_targets.mean(dim=0)
    val_targets = torch.stack([val_ds[i]["target"] for i in range(len(val_ds))])
    gm_dists = torch.sqrt(((val_targets - global_mean_xy) ** 2).sum(dim=1)).tolist()
    gm_matched = sum(d for d, m in zip(gm_dists, matched_mask) if m) / n_matched
    print(f"[global-mean] matched_subset_ADE={gm_matched:.4f}")

    scf_dists = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        d = torch.sqrt(((item["embeddings"][-1, -2:] - item["target"]) ** 2).sum())
        scf_dists.append(d.item())
    scf_matched = sum(d for d, m in zip(scf_dists, matched_mask) if m) / n_matched
    print(f"[static-copy-forward] matched_subset_ADE={scf_matched:.4f}\n")

    model_specs = [
        ("GRU", lambda: GRUHead(input_dim=input_dim)),
        ("ResidualGRU", lambda: ResidualGRUHead(input_dim=input_dim)),
        ("ResidualTransformer", lambda: ResidualTransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))),
    ]
    results, dists_by_model = {}, {}
    for name, ctor in model_specs:
        seed_everything(SEED)
        model, best_ade, best_epoch = train_one_model_weighted(name, ctor(), train_loader, val_loader, class_weights)
        dists = evaluate(model, val_loader)
        dists_by_model[name] = dists
        results[name] = sum(d for d, m in zip(dists, matched_mask) if m) / n_matched
        print(f"[{name}] matched_subset_ADE={results[name]:.4f}\n")

    print("=" * 70 + "\nSUMMARY (pooled, matched subset) -- class-balanced loss\n" + "=" * 70)
    print(f"  global-mean          : {gm_matched:.4f}")
    print(f"  static-copy-forward  : {scf_matched:.4f}")
    for name, ade in results.items():
        print(f"  {name:20s} : {ade:.4f}")

    print("\n" + "=" * 70 + "\nGAZE-TYPE BREAKDOWN -- class-balanced loss\n" + "=" * 70)
    for gtype in ("object", "social"):
        idx = [i for i in range(len(val_ds)) if matched_mask[i] and gaze_types[i] == gtype]
        if not idx:
            continue
        scf_sub = sum(scf_dists[i] for i in idx) / len(idx)
        print(f"\n  -- {gtype} (n={len(idx)}) -- static-copy-forward: {scf_sub:.4f}")
        for name in ("ResidualGRU", "ResidualTransformer"):
            model_sub = sum(dists_by_model[name][i] for i in idx) / len(idx)
            pct = (scf_sub - model_sub) / scf_sub * 100
            print(f"  {name:22s}: {model_sub:.4f} ({pct:+.1f}% vs static-copy-forward)")


if __name__ == "__main__":
    main()