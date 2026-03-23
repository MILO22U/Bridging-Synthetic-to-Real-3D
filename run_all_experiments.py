"""
Master runner — all experiments sequentially (one GPU task at a time).
Saves all outputs to visualizations2/ and writes results to md files.

Usage:
    python run_all_experiments.py
"""
import subprocess
import sys
import time
import os
import json
from datetime import datetime

LOG_FILE = "experiment_log.txt"
RESULTS_MD = "EXPERIMENT_RESULTS.md"

def log(msg):
    """Print and write to log file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def run(desc, cmd):
    log(f"\n{'='*60}")
    log(f"  {desc}")
    log(f"  CMD: {cmd}")
    log(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    log(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    elapsed = time.time() - t0
    # Write subprocess output to log
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-30:]:  # last 30 lines
            log(f"  | {line}")
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (rc={result.returncode})"
    log(f"\n>>> {desc}: {status} ({elapsed:.0f}s)")
    return ok, elapsed, result.stdout or ""

def save_experiment_md(name, ok, elapsed, output, save_dir, extra_info=""):
    """Save individual experiment results to md file."""
    md_path = os.path.join("visualizations2", f"{name.replace(' ', '_').replace('=', '')}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment: {name}\n\n")
        f.write(f"- **Status:** {'PASS' if ok else 'FAIL'}\n")
        f.write(f"- **Time:** {elapsed:.0f}s ({elapsed/60:.1f} min)\n")
        f.write(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- **Save dir:** `{save_dir}`\n")
        if extra_info:
            f.write(f"\n{extra_info}\n")
        # Count output files
        if os.path.isdir(save_dir):
            files = os.listdir(save_dir)
            f.write(f"- **Output files:** {len(files)}\n")
        f.write(f"\n## Last 20 lines of output\n```\n")
        for line in output.strip().split("\n")[-20:]:
            f.write(f"{line}\n")
        f.write("```\n")
    log(f"  Saved: {md_path}")

def update_summary_md(results):
    """Update the master EXPERIMENT_RESULTS.md."""
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("# Experiment Results — 2048-pt Model\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("| # | Experiment | Status | Time | Output Dir |\n")
        f.write("|---|-----------|--------|------|------------|\n")
        for i, (name, ok, elapsed, save_dir) in enumerate(results, 1):
            status = "PASS" if ok else "FAIL"
            f.write(f"| {i} | {name} | {status} | {elapsed:.0f}s | `{save_dir}` |\n")
        f.write("\n---\n\n")
        f.write("## Checkpoint\n")
        f.write("- Base model: `checkpoints/retrain_2048/best.pt` (val CD: 0.008105)\n")
        f.write("- Config: `config_2048.yaml` (2048 pts, 13 categories, ResNet-18)\n\n")

        # Check for DANN checkpoint
        dann_ckpt = "./checkpoints/dann_2048/best_model.pt"
        if os.path.exists(dann_ckpt):
            f.write(f"- DANN checkpoint: `{dann_ckpt}`\n\n")

        f.write("## Visualization Directories\n")
        f.write("All outputs in `visualizations2/`:\n")
        if os.path.isdir("visualizations2"):
            for d in sorted(os.listdir("visualizations2")):
                full = os.path.join("visualizations2", d)
                if os.path.isdir(full):
                    count = len([x for x in os.listdir(full) if not x.endswith('.md')])
                    f.write(f"- `{d}/` — {count} files\n")
                elif d.endswith('.md'):
                    f.write(f"- `{d}` (results)\n")
    log(f"\nUpdated {RESULTS_MD}")

def main():
    # Clear old log
    with open(LOG_FILE, "w") as f:
        f.write(f"Experiment run started: {datetime.now()}\n")

    os.makedirs("visualizations2", exist_ok=True)
    ckpt = "checkpoints/retrain_2048/best.pt"
    cfg = "config_2048.yaml"
    results = []  # (name, ok, elapsed, save_dir)

    # ─── 1. AdaIN (simple pixel-level) — 3 alpha values ───
    for alpha in [0.3, 0.5, 0.8]:
        save_dir = f"./visualizations2/adain_alpha{alpha}"
        ok, elapsed, output = run(
            f"AdaIN alpha={alpha}",
            f"python run_adain.py --checkpoint {ckpt} --config {cfg} "
            f"--photo_dir ./data/real_photos "
            f"--save_dir {save_dir} "
            f"--alpha {alpha}"
        )
        results.append((f"AdaIN alpha={alpha}", ok, elapsed, save_dir))
        save_experiment_md(
            f"AdaIN alpha={alpha}", ok, elapsed, output, save_dir,
            f"## Details\n- Style transfer alpha: {alpha}\n- Higher alpha = more synthetic style\n"
        )
        update_summary_md(results)

    # ─── 2. TTA on real photos ───
    save_dir = "./visualizations2/tta_real"
    ok, elapsed, output = run(
        "TTA on real photos",
        "python run_tta_real.py"
    )
    results.append(("TTA real photos", ok, elapsed, save_dir))
    # Extract spread stats from output
    spread_lines = [l for l in output.split("\n") if "spread" in l.lower()]
    extra = "## TTA Spread Results\n```\n" + "\n".join(spread_lines[-20:]) + "\n```\n" if spread_lines else ""
    save_experiment_md("TTA real photos", ok, elapsed, output, save_dir, extra)
    update_summary_md(results)

    # ─── 3. AdaIN VGG-level evaluation ───
    save_dir = "./visualizations2/adain_vgg"
    ok, elapsed, output = run(
        "AdaIN VGG evaluation (synthetic + real)",
        f"python eval_adain.py --config {cfg} --checkpoint {ckpt} "
        f"--output {save_dir}"
    )
    results.append(("AdaIN VGG eval", ok, elapsed, save_dir))
    # Extract CD metrics from output
    cd_lines = [l for l in output.split("\n") if "CD" in l.upper() or "cd" in l.lower()]
    extra = "## Metrics\n```\n" + "\n".join(cd_lines) + "\n```\n" if cd_lines else ""
    save_experiment_md("AdaIN VGG eval", ok, elapsed, output, save_dir, extra)
    update_summary_md(results)

    # ─── 4. DANN training ───
    save_dir = "./checkpoints/dann_2048"
    ok, elapsed, output = run(
        "DANN training (20 epochs, batch_size=8)",
        f"python train_dann.py --config {cfg} --checkpoint {ckpt} "
        f"--epochs 20 --batch-size 8"
    )
    results.append(("DANN training", ok, elapsed, save_dir))
    # Extract metrics
    metric_lines = [l for l in output.split("\n") if "CD" in l.upper() or "val_CD" in l or "Test" in l or "F@" in l or "epoch" in l.lower()]
    extra = "## Training Metrics\n```\n" + "\n".join(metric_lines[-25:]) + "\n```\n" if metric_lines else ""
    save_experiment_md("DANN training", ok, elapsed, output, save_dir, extra)
    update_summary_md(results)

    # ─── Final Summary ───
    log(f"\n{'='*60}")
    log("  ALL EXPERIMENTS COMPLETE")
    log(f"{'='*60}")
    total_time = sum(e for _, _, e, _ in results)
    for name, ok, elapsed, _ in results:
        log(f"  {'PASS' if ok else 'FAIL'}  {name} ({elapsed:.0f}s)")
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log(f"{'='*60}\n")

if __name__ == "__main__":
    main()
