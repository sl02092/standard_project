"""
train_temporal_poc_vacation.py — VACATION port of train_temporal_poc_ucolaeo.py.

WHAT'S IDENTICAL, COPIED VERBATIM: every model class, make_regression_tail/
make_residual_tail, count_params, pick_nhead, seed_everything, evaluate(),
train_one_model(), the resume-aware... (n/a here, no resume logic in this
file) global-mean/static-copy-forward baselines, the embeddings-zeroed
ablation, and the paired-bootstrap significance machinery added for
UCO-LAEO — all dataset-agnostic, none of it changes here either.

compute_baseline_computable_mask() again needed zero logic changes —
temporal_window_dataset_vacation.py's TemporalWindowDataset.rows carries
"clip", "subject", "last_context_path" under the same keys, and
compute_last_known_position_baseline_vacation.py's get_subject_gaze/
parse_frame_number match the established signature exactly. Only the
import line changed.

THE ONE REAL, NOT-JUST-MECHANICAL ADDITION: VACATION genuinely has both
gaze categories at real scale (social=93,450 / object=68,754 in the raw
population, confirmed against the static pipeline exactly) — unlike
UCO-LAEO, which structurally could only ever ask the pooled question.
This version reports a real social/object breakdown for every baseline
and every trained model (matched-subset ADE per type), plus a paired
bootstrap significance test per type for the PRIMARY (fair) criterion —
the same question VAT's own headline finding was built around, and the
first dataset since VAT itself where this project can ask it again with
real evidence rather than either ChildPlay's severe class imbalance or
UCO-LAEO's structural absence of the category. SANITY-criterion
bootstrap testing stays pooled-only, to keep scope bounded — the
PRIMARY breakdown is the one that actually matters for the object/social
question.

POPULATION SIZE NOTE: this dataset's population is substantially larger
than UCO-LAEO's (161,708 unique embedded frames vs. ~15,000) — expect
meaningfully longer wall-clock time per epoch. Worth a short calibration
run (POC_EPOCHS=10 or so) before committing to a longer one, same
instinct as UCO-LAEO's own epoch-budget lesson, just applied up front
here rather than discovered after the fact.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from temporal_window_dataset_vacation import (
    TemporalWindowDataset, build_clip_split, PRIMARY_HORIZON_MS, MANIFEST_PATH,
)
from compute_last_known_position_baseline_vacation import get_subject_gaze, parse_frame_number

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LENGTH = 8
# NOT hardcoded to 128 -- inferred from a real batch in main() before
# either model is constructed, same as every other dataset's version.

EPOCHS = int(os.environ.get("POC_EPOCHS", 40))
BATCH_SIZE = int(os.environ.get("POC_BATCH_SIZE", 64))
LR = float(os.environ.get("POC_LR", 1e-5))  # VAT's established value
SEED = 42


# ══════════════════════════════════════════════════════════════════════
# ── MODELS — copied verbatim, dataset-agnostic ──────────────────────────
# ══════════════════════════════════════════════════════════════════════

def make_regression_tail(input_dim, dropout=0.1):
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
# ── RESIDUAL VARIANTS — copied verbatim ─────────────────────────────────
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
# ── ABLATION: embeddings-zeroed control ──────────────────────────────────
# Direct test of whether a temporal head's real value comes from the
# 128-dim visual embeddings, or is achievable from the 2-dim predicted_xy
# coordinates alone (pure autoregressive drift). Wraps an EXISTING model
# rather than defining a new architecture.
# ══════════════════════════════════════════════════════════════════════

class EmbeddingZeroedWrapper(nn.Module):
    """Zeros the first `embed_dim` channels of each timestep's input
    (the visual embedding) before delegating to the wrapped model,
    leaving the last 2 channels (predicted_xy) untouched."""

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
    the top of the script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════
# ── PAIRED BOOTSTRAP SIGNIFICANCE — added 2026-07-31, see module docstring ──
# ══════════════════════════════════════════════════════════════════════

def matched_subset(dists, mask):
    """Filters a full-val-set list of per-row distances down to the
    baseline-computable (matched) subset, using the same mask everywhere
    else already uses. Small helper, exists so every bootstrap call site
    below does this identically rather than each re-writing the same
    list comprehension."""
    return [d for d, m in zip(dists, mask) if m]


def matched_subset_by_type(dists, mask, gaze_types, gtype):
    """Same as matched_subset(), further filtered to rows whose
    target_gaze_type == gtype. Added for VACATION specifically -- the
    first dataset since VAT itself where a real object/social breakdown
    is possible (see module docstring)."""
    return [d for d, m, gt in zip(dists, mask, gaze_types) if m and gt == gtype]


def paired_bootstrap_ci(a_dists, b_dists, n_resamples=10000, ci=0.95, seed=SEED):
    """Is mean(b_dists) - mean(a_dists) a real effect, or could it be
    noise from which particular validation rows happened to be sampled?

    a_dists, b_dists: same-length lists of per-row distances (normalized
    space), PAIRED row-for-row -- a_dists[i] and b_dists[i] must be
    errors for the SAME validation row (e.g. a model's prediction error
    and a baseline's prediction error against the identical target).
    Convention here: b is the thing being beaten (baseline), a is the
    thing doing the beating (model) -- observed_diff > 0 means a is
    better than b, matching how every "margin" is reported elsewhere in
    this script (baseline_ADE - model_ADE).

    Resamples n=len(a_dists) row-indices WITH REPLACEMENT, n_resamples
    times. Each resample uses the SAME drawn indices for both a and b
    (a paired bootstrap, not two independent ones) -- this preserves
    whatever correlation exists between a row's model error and its
    baseline error (both come from the same target, same context
    window), rather than artificially breaking that pairing.

    Returns (observed_diff, ci_low, ci_high). If ci_low > 0, the margin
    is robust across resampling -- not just true on this exact 398-row
    split. If the interval straddles 0, the observed margin could
    plausibly have gone the other way with a slightly different sample.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a_dists)
    b = np.asarray(b_dists)
    n = len(a)
    assert len(b) == n, (
        f"paired_bootstrap_ci needs equal-length, row-paired inputs, got "
        f"{n} vs {len(b)} -- these must be the SAME validation rows, not "
        f"just the same count by coincidence."
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
              f"be noise from this particular 398-row val split, not "
              f"necessarily a real effect)")
    elif not robust:
        print(f"    (entire interval is <= 0 -- no evidence of a real "
              f"positive margin here at all)")


# ══════════════════════════════════════════════════════════════════════
# ── BASELINE-COMPUTABLE MASK — UNCHANGED, only the import differs ──────
# ══════════════════════════════════════════════════════════════════════

def compute_baseline_computable_mask(dataset):
    """True for rows where the last-known-position (oracle) baseline
    could also produce an answer. Zero logic changes from ChildPlay's
    version -- see module docstring."""
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
    print(f"Horizon scope: target_horizon_ms={PRIMARY_HORIZON_MS} only, matching every other dataset's PoC scope.")
    print("Dataset note: VACATION has a real object-gaze category at scale "
          "(social=93,450 / object=68,754 in the raw population, confirmed "
          "against the static pipeline exactly) -- this PoC reports the "
          "full social/object breakdown, the same question VAT's own "
          "headline finding was built around.\n")

    train_ids, val_ids = build_clip_split(MANIFEST_PATH)
    train_ds = TemporalWindowDataset(split_clip_ids=train_ids, split_name="train")
    val_ds = TemporalWindowDataset(split_clip_ids=val_ids, split_name="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)  # order preserved -- required for the mask below

    input_dim = next(iter(train_loader))["embeddings"].shape[-1]
    print(f"Inferred per-frame input_dim = {input_dim} "
          f"({'128 embedding + 2 predicted_xy' if input_dim == 130 else 'embedding only, no predicted_xy'}).\n")

    print("Computing baseline-computable mask for the val set...")
    matched_mask = compute_baseline_computable_mask(val_ds)
    n_matched = sum(matched_mask)
    print(f"{n_matched}/{len(matched_mask)} val rows are baseline-computable "
          f"(matched subset for the fair comparison).\n")

    # target_gaze_type per val row, in the SAME order as everything else --
    # needed for the social/object breakdown below.
    val_gaze_types = [val_ds[i]["target_gaze_type"] for i in range(len(val_ds))]
    n_matched_social = sum(1 for m, gt in zip(matched_mask, val_gaze_types) if m and gt == "social")
    n_matched_object = sum(1 for m, gt in zip(matched_mask, val_gaze_types) if m and gt == "object")
    print(f"Matched subset composition: {n_matched_social} social, "
          f"{n_matched_object} object ({n_matched_social + n_matched_object} "
          f"of {n_matched} total -- any remainder has neither label, "
          f"shouldn't happen but checked rather than assumed).\n")

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
    gm_social = matched_subset_by_type(global_mean_dists, matched_mask, val_gaze_types, "social")
    gm_object = matched_subset_by_type(global_mean_dists, matched_mask, val_gaze_types, "object")
    print(f"    social (n={len(gm_social)}): ADE={sum(gm_social)/len(gm_social):.4f}  |  "
          f"object (n={len(gm_object)}): ADE={sum(gm_object)/len(gm_object):.4f}")
    print("  Any trained model scoring close to or worse than this is not "
          "trustworthy as genuine temporal signal.\n")

    results = {
        "global-mean (trivial)": {
            "full_val_ADE": global_mean_full_ade,
            "matched_subset_ADE": global_mean_matched_ade,
            "social_ADE": sum(gm_social) / len(gm_social),
            "object_ADE": sum(gm_object) / len(gm_object),
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
        cf_social = matched_subset_by_type(val_pred_xy_dists, matched_mask, val_gaze_types, "social")
        cf_object = matched_subset_by_type(val_pred_xy_dists, matched_mask, val_gaze_types, "object")
        print(f"    social (n={len(cf_social)}): ADE={sum(cf_social)/len(cf_social):.4f}  |  "
              f"object (n={len(cf_object)}): ADE={sum(cf_object)/len(cf_object):.4f}")
        print("  THIS is the fair floor a temporal model needs to beat. Compare "
              "against domain_transfer_check_vacation.py's own zero-horizon "
              "ADE_norm as a sanity cross-check -- the two are related but not "
              "identical (that script measures same-frame accuracy across the "
              "FULL cache; this measures it on the val split's final context "
              "frame specifically).\n")
        results["static-copy-forward (trivial)"] = {
            "full_val_ADE": copy_fwd_full_ade,
            "matched_subset_ADE": copy_fwd_matched_ade,
            "social_ADE": sum(cf_social) / len(cf_social),
            "object_ADE": sum(cf_object) / len(cf_object),
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
        model_specs.append((
            "ResidualGRU-coords-only (ablation)",
            lambda: EmbeddingZeroedWrapper(ResidualGRUHead(input_dim=input_dim), embed_dim=input_dim - 2)
        ))
    else:
        model_specs.append(("Transformer", lambda: TransformerHead(input_dim=input_dim, nhead=pick_nhead(input_dim))))
        print("[main] Residual variants and the embeddings-zeroed ablation "
              "SKIPPED -- both require predicted_xy in the input.\n")

    matched_dists_by_model = {}
    matched_dists_by_model_type = {}  # {name: {"social": [...], "object": [...]}}
    for name, constructor in model_specs:
        seed_everything(SEED)  # re-seed before EVERY phase, not just the first
        model = constructor()
        model, best_ade, best_epoch = train_one_model(name, model, train_loader, val_loader)
        dists = evaluate(model, val_loader)
        model_matched_dists = matched_subset(dists, matched_mask)
        matched_dists_by_model[name] = model_matched_dists
        matched_ade = sum(model_matched_dists) / n_matched

        model_social = matched_subset_by_type(dists, matched_mask, val_gaze_types, "social")
        model_object = matched_subset_by_type(dists, matched_mask, val_gaze_types, "object")
        matched_dists_by_model_type[name] = {"social": model_social, "object": model_object}

        results[name] = {
            "full_val_ADE": sum(dists) / len(dists),
            "matched_subset_ADE": matched_ade,
            "social_ADE": sum(model_social) / len(model_social) if model_social else float("nan"),
            "object_ADE": sum(model_object) / len(model_object) if model_object else float("nan"),
            "best_epoch": best_epoch + 1,
            "n_params": count_params(model),
        }

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY (VACATION, pooled + social/object breakdown)")
    print("=" * 78)
    print(f"{'Model':<26}{'params':<10}{'best_epoch':<12}{'full_val_ADE':<16}{'matched_ADE':<14}{'social_ADE':<13}{'object_ADE'}")
    for name, r in results.items():
        params_str = f"{r['n_params']:,}" if r['n_params'] else "-"
        print(f"{name:<26}{params_str:<10}{str(r['best_epoch']):<12}"
              f"{r['full_val_ADE']:<16.4f}{r['matched_subset_ADE']:<14.4f}"
              f"{r['social_ADE']:<13.4f}{r['object_ADE']:.4f}")

    print(f"\nn matched (baseline-computable) = {n_matched} / {len(matched_mask)} val rows "
          f"({n_matched_social} social, {n_matched_object} object)")
    print("\nNo oracle-ADE table exists for VACATION yet (unlike VAT/ChildPlay, "
          "whose baseline scripts print a full 'VAL-SPLIT ONLY' oracle table -- "
          "compute_last_known_position_baseline_vacation.py is a minimal lookup "
          "helper, not a report generator, same choice made for UCO-LAEO). "
          "The FAIR comparison below (static-copy-forward) doesn't need that "
          "table and is unaffected by its absence.")
    print("PRIMARY (fair) criterion: does any model beat static-copy-forward "
          "by a real margin on the matched subset -- and does that hold for "
          "BOTH gaze types, not just pooled?")
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
                  f"not dismissing.")
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
          f"set, not just in the one actual sample observed. This check "
          f"matters regardless of sample size, but matters most when n is "
          f"small -- with n={n_matched} here, judge for yourself against "
          f"UCO-LAEO's n=398 and ChildPlay's thousands whether that's a "
          f"real concern for this run specifically.\n")

    print("PRIMARY (fair): model vs. static-copy-forward --")
    if "static-copy-forward (trivial)" in results:
        for name in matched_dists_by_model:
            obs, lo, hi = paired_bootstrap_ci(
                matched_dists_by_model[name], copy_fwd_matched_dists
            )
            print_bootstrap_result(f"{name} (pooled)", obs, lo, hi)

            model_social = matched_dists_by_model_type[name]["social"]
            model_object = matched_dists_by_model_type[name]["object"]
            if model_social and cf_social:
                obs_s, lo_s, hi_s = paired_bootstrap_ci(model_social, cf_social)
                print_bootstrap_result(f"{name} (social)", obs_s, lo_s, hi_s)
            if model_object and cf_object:
                obs_o, lo_o, hi_o = paired_bootstrap_ci(model_object, cf_object)
                print_bootstrap_result(f"{name} (object)", obs_o, lo_o, hi_o)
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
              "reliable, not a coincidence of this val split.)")

    print(f"\nNOTE: LR defaults to {LR:.0e} here (VAT's established value). "
          f"Set POC_LR=1e-3 to deliberately replicate VAT/ChildPlay's original "
          f"starting-point progression instead, if wanted.")


if __name__ == "__main__":
    main()
