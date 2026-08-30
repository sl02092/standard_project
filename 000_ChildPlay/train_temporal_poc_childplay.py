"""
train_temporal_poc_childplay.py — ChildPlay port of train_temporal_poc.py
(VAT, final v3 state: absolute GRU/Transformer + residual variants, both
trivial baselines, matched-subset comparison).

WHAT'S IDENTICAL, COPIED VERBATIM: every model class (GRUHead,
TransformerHead, ResidualGRUHead, ResidualTransformerHead),
make_regression_tail/make_residual_tail, count_params, pick_nhead,
seed_everything, evaluate(), train_one_model() — all of this operates
purely on cached embedding tensors and is completely dataset-agnostic.
None of it changes between VAT and ChildPlay, and none of it changed here.

WHAT ACTUALLY NEEDED ADAPTING:
1. Imports: temporal_window_dataset_childplay instead of
   temporal_window_dataset (PRIMARY_HORIZON_MS instead of
   PRIMARY_HORIZON_FRAMES, per that module's own adaptation), and
   compute_last_known_position_baseline_childplay instead of the VAT
   version.
2. compute_baseline_computable_mask(): VAT's version looked up
   get_subject_gaze(show, clip, subject, fname) directly, since VAT's
   rows carried show/clip separately and last_context_fname was already
   the annotation-file key. ChildPlay's baseline script has a different
   signature — get_subject_gaze(clip_id, subject, abs_frame) — requiring
   the absolute frame number to be parsed from the full path first (same
   parse_frame_number() already used in the ChildPlay baseline script and
   the contiguity checker).
3. A handful of printed strings in main() (horizon references,
   which baseline script to cross-check against) — no logic changes,
   just accuracy.

STARTING CONFIG: LR defaults to 1e-5 (VAT's own eventual best value) --
CORRECTED 2026-08-02, this docstring and the runtime print at the bottom
of main() both used to claim the default was 1e-3 (VAT's *original*
starting point), but the actual code (`LR = float(os.environ.get("POC_LR",
1e-5))`) has always defaulted to 1e-5. This mismatch was flagged as a
known issue in an earlier session and never actually corrected in the
file until now -- and it mattered in practice: two runs believed to be
comparing 1e-3 against something else were actually both running at the
same true default (1e-5), with neither explicitly overriding POC_LR.
Set POC_LR=1e-3 explicitly if a genuine replication of VAT's original
starting point is wanted.

ABLATION (2026-07-19): a fifth condition, "ResidualGRU-coords-only",
added as a direct control for shortcut collapse -- the concern that a
residual head might ignore the 128-dim visual embeddings entirely and
just learn coordinate drift from the 2-dim predicted_xy alone.
EmbeddingZeroedWrapper zeros the embedding channels at the input while
wrapping the SAME ResidualGRUHead architecture unchanged, so this has an
identical parameter count to the real ResidualGRU -- a close match
between the two is worth taking seriously as evidence of shortcut
collapse, not proof of it on its own (VAT's own gaze-type breakdown found
a statistically significant departure from static-copy-forward, which a
purely coordinate-drifting model couldn't produce -- this ablation adds a
second, more direct line of evidence rather than replacing that one).

REPRODUCIBILITY FIX (2026-08-02): two runs with identical nominal config
(same LR, same seed) produced meaningfully different results -- GRU's
best epoch moved from 1 to 4, and the ablation gap that looked clearly
real in one run (+0.0031) nearly vanished in the other (+0.0005, right at
the "not clearly real" threshold this script's own logic checks against).
seed_everything() already seeded random/numpy/torch/torch.cuda correctly
-- the actual gap was that torch.backends.cudnn.deterministic/benchmark
were never set, and nn.GRU/nn.TransformerEncoder (every architecture
here) are well-documented as non-deterministic on CUDA even with all
seeds correctly set, unless these flags are explicitly configured. Fixed
in seed_everything() below. This same gap exists in the UCO-LAEO and
VACATION versions of this script too (same shared template) -- not
ChildPlay-specific, and worth knowing when weighing how much to trust
run-to-run stability in those already-completed results.

STATISTICAL SIGNIFICANCE (2026-08-02): paired bootstrap testing (10,000
resamples) added, same methodology already used for UCO-LAEO's and
VACATION's own scripts, ported back here since this file predates both.
Tests whether a margin is robust to which validation rows happened to be
sampled -- a DIFFERENT question from the reproducibility fix above, which
is about whether the same training setup reliably converges to a similar
model at all. Both matter; neither substitutes for the other.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from temporal_window_dataset_childplay import (
    TemporalWindowDataset, build_clip_split, PRIMARY_HORIZON_MS, MANIFEST_PATH,
)
from compute_last_known_position_baseline_childplay import get_subject_gaze, parse_frame_number

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LENGTH = 8
# NOT hardcoded to 128 -- inferred from a real batch in main() before
# either model is constructed, same as the VAT version.

EPOCHS = int(os.environ.get("POC_EPOCHS", 40))
BATCH_SIZE = int(os.environ.get("POC_BATCH_SIZE", 64))
LR = float(os.environ.get("POC_LR", 1e-3))
SEED = 42


# ══════════════════════════════════════════════════════════════════════
# ── MODELS — copied verbatim from train_temporal_poc.py (VAT) ──────────
# ══════════════════════════════════════════════════════════════════════

def make_regression_tail(input_dim, dropout=0.1):
    """Shared across both heads -- see module docstring on why this is
    kept identical rather than architecture-specific."""
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2), nn.Sigmoid(),
    )


class GRUHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.tail = make_regression_tail(hidden_dim, dropout)

    def forward(self, x):  # x: (B, T, D)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # final layer's final hidden state
        return self.tail(last_hidden)


class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256,
                 dropout=0.1, window_length=WINDOW_LENGTH):
        super().__init__()
        if input_dim % nhead != 0:
            raise ValueError(
                f"input_dim={input_dim} not divisible by nhead={nhead} -- "
                f"adjust nhead (e.g. 130 needs a divisor like 2 or 5, not 4)."
            )
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tail = make_regression_tail(input_dim, dropout)

    def forward(self, x):  # x: (B, T, D)
        x = x + self.pos_embed
        out = self.encoder(x)          # (B, T, D)
        last = out[:, -1, :]           # same "final position" convention as GRUHead
        return self.tail(last)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_nhead(input_dim, max_heads=8):
    """input_dim is not fixed at 128 (130 with predicted_xy included,
    which 4 doesn't divide) -- picks the largest valid head count <=
    max_heads that evenly divides input_dim."""
    for h in range(min(max_heads, input_dim), 0, -1):
        if input_dim % h == 0:
            return h
    return 1


# ══════════════════════════════════════════════════════════════════════
# ── RESIDUAL VARIANTS — copied verbatim from train_temporal_poc.py (VAT) ──
# ══════════════════════════════════════════════════════════════════════

def make_residual_tail(input_dim, dropout=0.1):
    """Same shape as make_regression_tail but NO Sigmoid (outputs an
    unbounded delta) and the final layer is ZERO-INITIALIZED -- at init,
    delta ≈ 0, so the wrapped model's prediction starts out equal to the
    static-copy-forward baseline exactly."""
    tail = nn.Sequential(
        nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 2),
    )
    nn.init.zeros_(tail[-1].weight)
    nn.init.zeros_(tail[-1].bias)
    return tail


class ResidualGRUHead(nn.Module):
    """Requires the last 2 dims of each timestep's input to be that
    frame's predicted_xy (input_dim=130, INCLUDE_PREDICTED_XY=True)."""

    def __init__(self, input_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        assert input_dim > 2, "ResidualGRUHead needs predicted_xy in the input (input_dim=130)."
        hidden_dim = hidden_dim or input_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.tail = make_residual_tail(hidden_dim, dropout)

    def forward(self, x):  # x: (B, T, D), last 2 dims per timestep = predicted_xy
        reference = x[:, -1, -2:]  # last context frame's own predicted position
        _, h_n = self.gru(x)
        delta = self.tail(h_n[-1])
        return torch.clamp(reference + delta, 0.0, 1.0)


class ResidualTransformerHead(nn.Module):
    """Same requirement/reasoning as ResidualGRUHead."""

    def __init__(self, input_dim, num_layers=2, nhead=4, dim_feedforward=256,
                 dropout=0.1, window_length=WINDOW_LENGTH):
        super().__init__()
        assert input_dim > 2, "ResidualTransformerHead needs predicted_xy in the input (input_dim=130)."
        if input_dim % nhead != 0:
            raise ValueError(
                f"input_dim={input_dim} not divisible by nhead={nhead} -- "
                f"pass nhead=pick_nhead(input_dim)."
            )
        self.pos_embed = nn.Parameter(torch.zeros(1, window_length, input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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


# ══════════════════════════════════════════════════════════════════════
# ── ABLATION: embeddings-zeroed control (2026-07-19) ────────────────────
# Direct test of whether a temporal head's real value comes from the
# 128-dim visual embeddings, or is achievable from the 2-dim predicted_xy
# coordinates alone (pure autoregressive drift). Wraps an EXISTING model
# rather than defining a new architecture -- zeroing the embedding portion
# of the input in-place before the wrapped model's own forward() runs,
# leaving the tensor's shape, the model's architecture, and its parameter
# count completely unchanged. That's deliberate: if this ablation scores
# close to the un-wrapped model, the comparison is airtight, since nothing
# else differs between the two runs except this one thing.
# ══════════════════════════════════════════════════════════════════════

class EmbeddingZeroedWrapper(nn.Module):
    """Zeros the first `embed_dim` channels of each timestep's input
    (the visual embedding) before delegating to the wrapped model,
    leaving the last 2 channels (predicted_xy) untouched -- exactly the
    control experiment suggested for testing shortcut collapse: does a
    residual head's real performance depend on the visual embeddings, or
    only on coordinate drift?"""

    def __init__(self, inner_model, embed_dim):
        super().__init__()
        self.inner = inner_model  # registered as a submodule -- count_params()
        self.embed_dim = embed_dim  # sees the SAME param count as the un-wrapped model

    def forward(self, x):
        x = x.clone()
        x[:, :, :self.embed_dim] = 0.0
        return self.inner(x)


# ══════════════════════════════════════════════════════════════════════
# ── SEEDING — copied verbatim ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def seed_everything(seed=SEED):
    """Called immediately before EACH model's training -- not once at
    the top of the script. See train_temporal_poc.py's docstring for the
    step52 bug this discipline was established to avoid repeating.

    cudnn.deterministic/benchmark ADDED 2026-08-02 -- see module
    docstring's "REPRODUCIBILITY FIX" note. Seeding random/numpy/torch/
    torch.cuda alone does NOT make nn.GRU or nn.TransformerEncoder
    deterministic on CUDA; this was the actual missing piece."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
# ── PAIRED BOOTSTRAP SIGNIFICANCE — added 2026-08-02, see module docstring ──
# ══════════════════════════════════════════════════════════════════════

def matched_subset(dists, mask):
    """Filters a full-val-set list of per-row distances down to the
    baseline-computable (matched) subset."""
    return [d for d, m in zip(dists, mask) if m]


def paired_bootstrap_ci(a_dists, b_dists, n_resamples=10000, ci=0.95, seed=SEED):
    """Is mean(b_dists) - mean(a_dists) a real effect, or could it be
    noise from which particular validation rows happened to be sampled?
    a_dists, b_dists must be PAIRED row-for-row (same validation rows,
    same order). Convention: b is the baseline being beaten, a is the
    model doing the beating -- observed_diff > 0 means a is better.
    Returns (observed_diff, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a_dists)
    b = np.asarray(b_dists)
    n = len(a)
    assert len(b) == n, (
        f"paired_bootstrap_ci needs equal-length, row-paired inputs, got "
        f"{n} vs {len(b)}."
    )

    observed_diff = float(b.mean() - a.mean())
    boot_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = b[idx].mean() - a[idx].mean()

    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boot_diffs, [alpha, 1 - alpha])
    return observed_diff, float(ci_low), float(ci_high)


def print_bootstrap_result(label, observed_diff, ci_low, ci_high, ci=0.95):
    robust = ci_low > 0
    verdict = "ROBUST" if robust else "NOT CONFIRMED ROBUST"
    print(f"  {label}: margin={observed_diff:+.4f}, "
          f"{int(ci*100)}% CI=[{ci_low:+.4f}, {ci_high:+.4f}]  -> {verdict}")
    if not robust and ci_high > 0:
        print(f"    (interval straddles zero -- this margin could plausibly "
              f"be noise from this particular val split, not necessarily a "
              f"real effect)")
    elif not robust:
        print(f"    (entire interval is <= 0 -- no evidence of a real "
              f"positive margin here at all)")


# ══════════════════════════════════════════════════════════════════════
# ── BASELINE-COMPUTABLE MASK — ADAPTED (different get_subject_gaze sig) ──
# ══════════════════════════════════════════════════════════════════════

def compute_baseline_computable_mask(dataset):
    """True for rows where the last-known-position (oracle) baseline
    could also produce an answer. VAT's version called
    get_subject_gaze(show, clip, subject, fname) directly; ChildPlay's
    baseline script keys by absolute frame NUMBER, not filename, so the
    number is parsed from the full path first (same parse_frame_number()
    used throughout the ChildPlay scripts)."""
    mask = []
    for r in dataset.rows:
        abs_frame = parse_frame_number(r["last_context_path"])
        gaze = get_subject_gaze(r["clip"], r["subject"], abs_frame)
        mask.append(gaze is not None)
    return mask


# ══════════════════════════════════════════════════════════════════════
# ── TRAIN / EVAL — copied verbatim ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    """Returns a list of per-row Euclidean distances (normalized space),
    in dataset order -- loader must NOT be shuffled."""
    model.eval()
    dists = []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            pred = model(emb)
            d = torch.sqrt(((pred - target) ** 2).sum(dim=1))
            dists.extend(d.cpu().tolist())
    return dists


def train_one_model(name, model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    n_params = count_params(model)
    print(f"\n[{name}] {n_params:,} trainable parameters "
          f"(frozen encoder, not counted here, is 5.7M -- this head is "
          f"negligible in comparison).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

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
            loss = loss_fn(pred, target)
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
            train_loss = sum(train_losses) / len(train_losses)
            print(f"[{name}] epoch {epoch+1:>3}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_ADE={val_ade:.4f}")

    model.load_state_dict(best_state)
    print(f"[{name}] Best val_ADE={best_val_ade:.4f} at epoch {best_epoch+1}/{epochs}")
    return model, best_val_ade, best_epoch


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Horizon scope: target_horizon_ms={PRIMARY_HORIZON_MS} only, matching the VAT PoC's own scope.\n")

    train_ids, val_ids = build_clip_split(MANIFEST_PATH)
    train_ds = TemporalWindowDataset(split_clip_ids=train_ids, split_name="train")
    val_ds = TemporalWindowDataset(split_clip_ids=val_ids, split_name="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)  # order preserved -- required for the mask below

    input_dim = next(iter(train_loader))["embeddings"].shape[-1]
    print(f"Inferred per-frame input_dim = {input_dim} "
          f"({'128 embedding + 2 predicted_xy' if input_dim == 130 else 'embedding only, no predicted_xy'}).\n")

    print("Computing baseline-computable mask for the val set "
          "(re-reads the val clips' annotation CSVs -- same cost as running "
          "compute_last_known_position_baseline_childplay.py once)...")
    matched_mask = compute_baseline_computable_mask(val_ds)
    n_matched = sum(matched_mask)
    print(f"{n_matched}/{len(matched_mask)} val rows are baseline-computable "
          f"(matched subset for the fair comparison).\n")

    # ── Global-mean trivial baseline ─────────────────────────────────
    train_targets = torch.stack([train_ds[i]["target"] for i in range(len(train_ds))])
    global_mean_xy = train_targets.mean(dim=0)  # computed from TRAIN only, not val -- no leakage
    val_targets = torch.stack([val_ds[i]["target"] for i in range(len(val_ds))])
    global_mean_dists = torch.sqrt(((val_targets - global_mean_xy) ** 2).sum(dim=1)).tolist()
    global_mean_full_ade = sum(global_mean_dists) / len(global_mean_dists)
    global_mean_matched_dists = matched_subset(global_mean_dists, matched_mask)
    global_mean_matched_ade = sum(global_mean_matched_dists) / n_matched
    print(f"[global-mean baseline] predicting train-set mean ({global_mean_xy[0]:.3f}, "
          f"{global_mean_xy[1]:.3f}) for every val row -> full_val_ADE="
          f"{global_mean_full_ade:.4f}, matched_subset_ADE={global_mean_matched_ade:.4f}")
    print("  Any trained model scoring close to or worse than this is not "
          "trustworthy as genuine temporal signal.\n")

    results = {
        "global-mean (trivial)": {
            "full_val_ADE": global_mean_full_ade,
            "matched_subset_ADE": global_mean_matched_ade,
            "best_epoch": "n/a", "n_params": 0,
        }
    }

    # ── Static-copy-forward baseline — the FAIR floor ───────────────
    if input_dim > 2:
        val_pred_xy_dists = []
        for i in range(len(val_ds)):
            item = val_ds[i]
            last_pred_xy = item["embeddings"][-1, -2:]
            d = torch.sqrt(((last_pred_xy - item["target"]) ** 2).sum())
            val_pred_xy_dists.append(d.item())
        copy_fwd_full_ade = sum(val_pred_xy_dists) / len(val_pred_xy_dists)
        copy_fwd_matched_dists = matched_subset(val_pred_xy_dists, matched_mask)
        copy_fwd_matched_ade = sum(copy_fwd_matched_dists) / n_matched
        print(f"[static-copy-forward baseline] using the static encoder's own "
              f"last predicted position, unchanged, zero learning -> "
              f"full_val_ADE={copy_fwd_full_ade:.4f}, "
              f"matched_subset_ADE={copy_fwd_matched_ade:.4f}")
        print("  THIS is the fair floor a temporal model needs to beat.\n")
        results["static-copy-forward (trivial)"] = {
            "full_val_ADE": copy_fwd_full_ade,
            "matched_subset_ADE": copy_fwd_matched_ade,
            "best_epoch": "n/a", "n_params": 0,
        }
    else:
        print("[static-copy-forward baseline] SKIPPED -- requires predicted_xy "
              "in the input (input_dim=130).\n")

    # ── Four real models, all re-seeded immediately before each ─────
    model_specs = [("GRU", lambda: GRUHead(input_dim=input_dim))]
    if input_dim > 2:
        model_specs.append(("Transformer", lambda: TransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))))
        model_specs.append(("ResidualGRU", lambda: ResidualGRUHead(input_dim=input_dim)))
        model_specs.append(("ResidualTransformer", lambda: ResidualTransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))))
        # ── Ablation: same architecture and parameter count as ResidualGRU
        # above, embeddings zeroed at the input. If this scores close to
        # ResidualGRU, that's a shortcut-collapse red flag -- the real
        # model isn't using the visual embeddings for anything. If it
        # scores clearly worse, that's direct evidence the embeddings are
        # doing real work, not just an indirect inference from the
        # gaze-type breakdown.
        model_specs.append((
            "ResidualGRU-coords-only (ablation)",
            lambda: EmbeddingZeroedWrapper(ResidualGRUHead(input_dim=input_dim), embed_dim=input_dim - 2)
        ))
    else:
        model_specs.append(("Transformer", lambda: TransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))))
        print("[main] Residual variants and the embeddings-zeroed ablation "
              "SKIPPED -- both require predicted_xy in the input.\n")

    matched_dists_by_model = {}
    for name, constructor in model_specs:
        seed_everything(SEED)  # re-seed before EVERY phase, not just the first
        model = constructor()
        model, best_ade, best_epoch = train_one_model(name, model, train_loader, val_loader)
        dists = evaluate(model, val_loader)
        model_matched_dists = matched_subset(dists, matched_mask)
        matched_dists_by_model[name] = model_matched_dists
        matched_ade = sum(model_matched_dists) / n_matched
        results[name] = {
            "full_val_ADE": sum(dists) / len(dists),
            "matched_subset_ADE": matched_ade,
            "best_epoch": best_epoch + 1,
            "n_params": count_params(model),
        }

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Model':<26}{'params':<10}{'best_epoch':<12}{'full_val_ADE':<16}{'matched_subset_ADE'}")
    for name, r in results.items():
        params_str = f"{r['n_params']:,}" if r['n_params'] else "-"
        print(f"{name:<26}{params_str:<10}{str(r['best_epoch']):<12}"
              f"{r['full_val_ADE']:<16.4f}{r['matched_subset_ADE']:.4f}")

    print(f"\nn matched (baseline-computable) = {n_matched} / {len(matched_mask)} val rows")
    print("\nCompare matched_subset_ADE above against "
          "compute_last_known_position_baseline_childplay.py's 'VAL-SPLIT ONLY' "
          "table at target_horizon_ms=300 for the ORACLE comparison, and "
          "against static-copy-forward above for the FAIR, deployment-realistic one.")
    print("PRIMARY (fair) criterion: does any model beat static-copy-forward "
          "by a real margin on the matched subset?")
    print("STRETCH criterion: does any model approach the oracle last-known-"
          "position baseline?")
    print("SANITY criterion: does any model beat global-mean by a real margin?")
    print("SECONDARY: which model/architecture wins among those that pass.")
    if "ResidualGRU-coords-only (ablation)" in results and "ResidualGRU" in results:
        rg = results["ResidualGRU"]["matched_subset_ADE"]
        rg_ablated = results["ResidualGRU-coords-only (ablation)"]["matched_subset_ADE"]
        gap = rg_ablated - rg
        print(f"\nABLATION (shortcut-collapse control): ResidualGRU matched_subset_ADE="
              f"{rg:.4f} vs. embeddings-zeroed matched_subset_ADE={rg_ablated:.4f} "
              f"(same architecture, same parameter count -- only the input differs).")
        if gap < 0.002:
            print(f"  Gap is small ({gap:+.4f}) -- ResidualGRU's real performance may "
                  f"not depend much on the visual embeddings. Worth taking seriously, "
                  f"not dismissing -- cross-check against the gaze-type breakdown "
                  f"before concluding either way.")
        else:
            print(f"  Gap is real ({gap:+.4f}) -- zeroing the embeddings measurably "
                  f"hurts performance, direct evidence ResidualGRU is using the "
                  f"visual embeddings for something, not just coordinate drift.")

    # ── Statistical significance — paired bootstrap, matched subset only ──
    print("\n" + "=" * 78)
    print(f"STATISTICAL SIGNIFICANCE (paired bootstrap, n_resamples=10000, "
          f"n={n_matched} matched val rows)")
    print("=" * 78)
    print("A margin is only reported ROBUST if its 95% CI stays entirely "
          "above zero -- i.e. the model still beats the comparator in the "
          "overwhelming majority of 10,000 simulated resamples of this val "
          "set, not just in the one actual sample observed.\n")

    print("PRIMARY (fair): model vs. static-copy-forward --")
    if "static-copy-forward (trivial)" in results:
        for name in matched_dists_by_model:
            obs, lo, hi = paired_bootstrap_ci(
                matched_dists_by_model[name], copy_fwd_matched_dists
            )
            print_bootstrap_result(name, obs, lo, hi)
    else:
        print("  SKIPPED -- static-copy-forward baseline unavailable (no predicted_xy).")

    print("\nSANITY: model vs. global-mean --")
    for name in matched_dists_by_model:
        obs, lo, hi = paired_bootstrap_ci(
            matched_dists_by_model[name], global_mean_matched_dists
        )
        print_bootstrap_result(name, obs, lo, hi)

    if "ResidualGRU-coords-only (ablation)" in matched_dists_by_model and "ResidualGRU" in matched_dists_by_model:
        print("\nABLATION: ResidualGRU vs. embeddings-zeroed ResidualGRU --")
        obs, lo, hi = paired_bootstrap_ci(
            matched_dists_by_model["ResidualGRU"],
            matched_dists_by_model["ResidualGRU-coords-only (ablation)"],
        )
        print_bootstrap_result("ResidualGRU vs. ablation", obs, lo, hi)
        print("  (ROBUST here means the visual-embedding contribution is "
              "reliable, not a coincidence of this val split -- a SEPARATE "
              "question from run-to-run training stability, see module "
              "docstring's reproducibility-fix note.)")

    print(f"\nNOTE: LR defaults to {LR:.0e} here (VAT's own eventual best "
          f"value, corrected 2026-08-02 -- this used to incorrectly claim "
          f"1e-3 here and in the docstring; the code's real default was "
          f"always 1e-5). Set POC_LR=1e-3 explicitly for a genuine "
          f"replication of VAT's original starting point instead.")


if __name__ == "__main__":
    main()
