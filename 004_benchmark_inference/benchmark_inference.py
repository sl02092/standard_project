"""
benchmark_inference.py — latency / FPS / VRAM benchmarking for the static
student and all 4 temporal head architectures, alone and combined into
the real end-to-end anticipation pipeline.

RUN THIS LOCALLY, NOT ON HPC. HPC's GPU is a 3g.40gb slice of a
data-center A100 -- not remotely representative of hardware an actual
robot would carry. Your own machine (GPU and CPU both) gives a far more
honest "is this deployable" number for a human-robot interaction thesis.

METHODOLOGY NOTES -- read before trusting the numbers:
- CUDA ops are asynchronous. Timing without torch.cuda.synchronize()
  measures how fast Python can ISSUE work, not how fast the GPU actually
  finishes it -- a common, silent way to get numbers that look great and
  mean nothing. Every timed region here synchronizes both before starting
  the clock and after the forward pass completes.
- The first several forward passes are slower (CUDA context / kernel
  warm-up, cuDNN autotuning) -- discarded, not included in reported stats.
- Reports mean + std across many repeats, not one measurement -- GPU
  timing has real run-to-run noise.
- Batch size 1 throughout -- matches real deployment (one live camera
  frame at a time), not offline batch throughput.
- No trained checkpoints needed. Latency/VRAM depend only on
  architecture SHAPE, not which teacher trained the weights -- random
  initialization has identical compute cost to a real trained model. So
  this only needs 2 static sizes x 4 temporal architectures, not the
  full 8-condition grid.
- Reports TWO different "anticipation latency" numbers, not one:
    cold_start  = 8 static passes (filling the context buffer) + 1
                  temporal pass -- the one-time cost before the very
                  first anticipation output.
    steady_state = 1 static pass (the newest frame) + 1 temporal pass --
                  the real, sustained per-frame cost once a live system
                  is already running with a full buffer. THIS is the
                  number that matters for "can this run in real time."
"""

import os
import time
import argparse
import statistics

import torch
import torch.nn as nn

from gaze_student_model import GazeStudent, IMG_SIZE, HEAD_CROP_SIZE, count_params

WINDOW_LENGTH = 8
N_WARMUP = 10
N_TIMED = 100


# ══════════════════════════════════════════════════════════════════════
# ── TEMPORAL MODEL LAYER — copied verbatim from train_temporal_v1.py
# ── (same source, same reasoning as eval_temporal_v1.py's own copy) ────
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


TEMPORAL_ARCHS = {
    "GRU": GRUHead, "Transformer": TransformerHead,
    "ResidualGRU": ResidualGRUHead, "ResidualTransformer": ResidualTransformerHead,
}


# ══════════════════════════════════════════════════════════════════════
# ── TIMING HARNESS ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def time_forward(fn, device, n_warmup=N_WARMUP, n_timed=N_TIMED):
    """Runs fn() n_warmup+n_timed times, properly synchronized. Returns
    (mean_ms, std_ms) over the n_timed runs, warm-up runs discarded."""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_timed):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    return statistics.mean(times_ms), statistics.stdev(times_ms)


def measure_vram(fn, device):
    """Peak VRAM in MB during fn(). Returns None on CPU (no meaningful
    equivalent -- system RAM isn't tracked the same way)."""
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         choices=["cuda", "cpu"])
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}\n")

    results = []

    # ── Static models, both sizes ──────────────────────────────────
    static_configs = [("ViT-Tiny", "vit_tiny_patch16_224"), ("ViT-Small", "vit_small_patch16_224")]
    static_timings = {}  # label -> (mean_ms, std_ms) for reuse in combined pipeline numbers

    for label, vit_model in static_configs:
        model = GazeStudent(vit_model=vit_model).to(device).eval()
        n_params = count_params(model)

        scene = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        head = torch.randn(1, 3, HEAD_CROP_SIZE, HEAD_CROP_SIZE, device=device)

        with torch.no_grad():
            fn = lambda: model(scene, head)
            mean_ms, std_ms = time_forward(fn, device)
            vram_mb = measure_vram(fn, device)

        static_timings[label] = (mean_ms, std_ms)
        results.append({
            "component": f"Static [{label}]", "params": n_params,
            "latency_ms": mean_ms, "latency_std_ms": std_ms,
            "fps": 1000 / mean_ms, "vram_mb": vram_mb,
        })
        vram_str = f", VRAM={vram_mb:.1f}MB" if vram_mb else ""
        print(f"Static [{label}]: {n_params/1e6:.1f}M params, "
              f"{mean_ms:.2f}±{std_ms:.2f}ms, {1000/mean_ms:.1f} FPS{vram_str}")

    # ── Temporal heads, all 4 architectures (input_dim=130 is fixed --
    # 128-dim embedding + 2-dim predicted_xy -- independent of which
    # static model size produced it) ───────────────────────────────
    input_dim = 130
    temporal_timings = {}

    for name, cls in TEMPORAL_ARCHS.items():
        if name in ("Transformer", "ResidualTransformer"):
            model = cls(input_dim=input_dim, nhead=pick_nhead(input_dim)).to(device).eval()
        else:
            model = cls(input_dim=input_dim).to(device).eval()
        n_params = count_params(model)

        seq = torch.randn(1, WINDOW_LENGTH, input_dim, device=device)

        with torch.no_grad():
            fn = lambda: model(seq)
            mean_ms, std_ms = time_forward(fn, device)
            vram_mb = measure_vram(fn, device)

        temporal_timings[name] = (mean_ms, std_ms)
        results.append({
            "component": f"Temporal [{name}]", "params": n_params,
            "latency_ms": mean_ms, "latency_std_ms": std_ms,
            "fps": 1000 / mean_ms, "vram_mb": vram_mb,
        })
        vram_str = f", VRAM={vram_mb:.1f}MB" if vram_mb else ""
        print(f"Temporal [{name}]: {n_params/1e3:.0f}K params, "
              f"{mean_ms:.2f}±{std_ms:.2f}ms, {1000/mean_ms:.1f} FPS{vram_str}")

    # ── Combined pipeline numbers, both static sizes x all 4 temporal
    # architectures -- cold-start and steady-state, see module docstring ──
    print("\n--- Combined pipeline (static + temporal) ---")
    for static_label, (static_mean, _) in static_timings.items():
        for temporal_name, (temporal_mean, _) in temporal_timings.items():
            cold_start_ms = 8 * static_mean + temporal_mean
            steady_state_ms = static_mean + temporal_mean
            results.append({
                "component": f"Pipeline [{static_label} + {temporal_name}]",
                "params": None,
                "cold_start_ms": cold_start_ms, "steady_state_ms": steady_state_ms,
                "steady_state_fps": 1000 / steady_state_ms,
            })
            print(f"  {static_label} + {temporal_name}: "
                  f"cold_start={cold_start_ms:.2f}ms  "
                  f"steady_state={steady_state_ms:.2f}ms ({1000/steady_state_ms:.1f} FPS)")

    # ── Save ─────────────────────────────────────────────────────────
    import json
    out_dir = os.environ.get("BENCHMARK_OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"benchmark_{device.type}.json")
    with open(out_path, "w") as f:
        json.dump({"device": str(device), "n_warmup": N_WARMUP, "n_timed": N_TIMED,
                    "results": results}, f, indent=2, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
