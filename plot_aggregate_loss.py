"""Plot aggregate training loss across all workers.

Loads per-rank training_metrics JSON files, sums the loss at each step
across workers, and saves the resulting loss curve.

Usage:
    python plot_aggregate_loss.py --output_dir RTE --reduce_type gather
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot aggregate loss across workers")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing per-rank metrics files")
    parser.add_argument("--reduce_type", type=str, required=True, help="Reduce type used in filenames")
    args = parser.parse_args()

    pattern = os.path.join(args.output_dir, f"training_metrics_{args.reduce_type}_rank*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching {pattern}")
        return

    # Load per-rank step losses
    all_rank_losses = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        all_rank_losses.append([s["loss"] for s in data["step_losses"]])
        print(f"Loaded {f} ({len(all_rank_losses[-1])} steps)")

    # Sum losses across ranks at each step
    min_steps = min(len(losses) for losses in all_rank_losses)
    summed_losses = [
        sum(rank_losses[i] for rank_losses in all_rank_losses)
        for i in range(min_steps)
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summed_losses, alpha=0.7, label=f"summed loss ({len(files)} workers)")
    for rank_idx, rank_losses in enumerate(all_rank_losses):
        ax.plot(rank_losses[:min_steps], alpha=0.3, linewidth=0.8, label=f"rank {rank_idx}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Aggregate Training Loss ({args.reduce_type}, {len(files)} workers)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, f"loss_curve_{args.reduce_type}_aggregate.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
