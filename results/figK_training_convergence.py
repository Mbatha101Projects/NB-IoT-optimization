#!/usr/bin/env python3
"""
figK_training_convergence.py — Training convergence evidence

Reads training_log.csv (produced by train.py) and plots reward
progression and epsilon decay across the 60,000-episode training run.
This is the direct visual evidence backing Section IV-D's convergence
claims, which currently only describe the process in text.

Expects columns: episode, total_reward, epsilon, min_visit_count,
table_coverage_pct — matching train.py's actual logged output.

Run this in the same folder as your real training_log.csv.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LOG_PATH = os.path.join(SCRIPT_DIR, "training_log.csv")
if not os.path.exists(LOG_PATH):
    raise FileNotFoundError(
        f"training_log.csv not found at {LOG_PATH}. "
        "Copy your actual training log into this folder before running."
    )

df = pd.read_csv(LOG_PATH)
print(f"[INFO] Loaded {len(df)} episodes from training_log.csv")

# Smooth the reward curve with a rolling mean -- raw per-episode reward
# is noisy episode-to-episode (random condition changes mid-episode),
# the rolling mean is what actually shows the convergence trend clearly
WINDOW = max(1, len(df) // 200)
df["reward_smoothed"] = df["total_reward"].rolling(WINDOW, min_periods=1).mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(df["episode"], df["total_reward"], color="#B0BEC5", linewidth=0.5, alpha=0.6, label="Raw per-episode")
axes[0].plot(df["episode"], df["reward_smoothed"], color="#1565C0", linewidth=1.8, label=f"Rolling mean (window={WINDOW})")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].set_title("Reward Progression")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["episode"], df["epsilon"], color="#EF6C00", linewidth=1.5)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Epsilon (exploration rate)")
axes[1].set_title("Exploration Decay")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0.05, color="gray", linestyle="--", linewidth=1, label="Floor (0.05)")
axes[1].legend(fontsize=8)

fig.suptitle("Training Convergence — 60,000 Episodes")
fig.tight_layout()
path = os.path.join(FIG_DIR, "figK_training_convergence.png")
fig.savefig(path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"[SAVED] {path}")
