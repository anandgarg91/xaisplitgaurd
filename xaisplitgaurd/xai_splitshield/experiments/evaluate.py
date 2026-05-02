"""
experiments/evaluate.py
────────────────────────
Standalone evaluation script — loads a trained checkpoint and computes
ASR, Clean Accuracy, and detection metrics across all attack types.

Usage:
    python experiments/evaluate.py --config configs/cifar10_badnets.yaml
    python experiments/evaluate.py \\
        --checkpoint results/cifar10_badnets_xai_splitshield_split2_s42/checkpoints \\
        --dataset cifar10 --attack all
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from data.dataset_loader import get_dataset
from data.poisoning import get_attack
from models.resnet import build_split_resnet18, get_smashed_shape
from defense.xai_splitshield import XAISplitShield
from utils.metrics import compute_asr, compute_clean_accuracy
from utils.logger import get_logger
from utils.visualization import plot_asr_ca_comparison

ATTACKS = ["badnets", "blended", "wanet", "lira"]
DEFENSES = ["none", "xai_splitshield"]

logger = get_logger("evaluate")


def evaluate_checkpoint(cfg: dict, ckpt_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, num_classes = get_dataset(
        name=cfg["dataset"],
        root=cfg.get("root", "./data/raw"),
        image_size=cfg.get("image_size", 32),
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 4),
    )

    client_model, server_model = build_split_resnet18(
        num_classes=num_classes,
        split_layer=cfg.get("split_layer", 2),
    )
    ckpt_dir = Path(ckpt_dir)
    client_model.load_state_dict(
        torch.load(ckpt_dir / "client_model.pth", map_location=device)
    )
    server_model.load_state_dict(
        torch.load(ckpt_dir / "server_model.pth", map_location=device)
    )
    client_model = client_model.to(device)
    server_model = server_model.to(device)
    client_model.eval()
    server_model.eval()

    smashed_shape = get_smashed_shape(client_model, cfg.get("image_size", 32), device=device)

    results = {}

    # ── Evaluate each attack ──────────────────────────────────────────────────
    attack_names = ATTACKS if cfg.get("attack") == "all" else [cfg["attack"]]

    for atk_name in attack_names:
        attack = get_attack(
            name=atk_name,
            cfg=cfg,
            image_size=cfg.get("image_size", 32),
            num_classes=num_classes,
            device=device,
        )

        for def_name in DEFENSES:
            shield = None
            if def_name == "xai_splitshield":
                shield = XAISplitShield(
                    server_model=server_model,
                    smashed_shape=tuple(smashed_shape),
                    device=device,
                    cfg=cfg,
                )
                # Run a short warm-up from test loader (clean samples only)
                logger.info("Running evaluation warm-up...")
                for images, labels in test_loader:
                    shield.warmup(
                        client_model(images.to(device)).detach(),
                        labels.to(device)
                    )
                shield.calibrate()

            ca  = compute_clean_accuracy(client_model, server_model,
                                         test_loader, device, shield)
            asr = compute_asr(client_model, server_model, test_loader,
                              attack, cfg.get("target_class", 0), device, shield)

            key = f"{atk_name}_{def_name}"
            results[key] = {"asr": asr, "ca": ca}
            logger.info(
                f"[{atk_name.upper()}][{def_name}]  "
                f"CA={ca*100:.2f}%  ASR={asr*100:.2f}%"
            )

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_dir = Path(ckpt_dir).parent / "eval_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_asr_ca_comparison(
        {k: v for k, v in results.items()},
        metric="asr",
        title=f"ASR Comparison — {cfg['dataset'].upper()}",
        save_path=str(out_dir / "asr_comparison.png"),
    )
    logger.info(f"Evaluation plots saved to {out_dir}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint directory containing "
                        "client_model.pth and server_model.pth")
    p.add_argument("--dataset",    type=str, default=None)
    p.add_argument("--attack",     type=str, default="all")
    args = p.parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        for section in raw.values():
            if isinstance(section, dict):
                cfg.update(section)

    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.attack:
        cfg["attack"] = args.attack

    cfg.setdefault("dataset",      "cifar10")
    cfg.setdefault("attack",       "all")
    cfg.setdefault("target_class", 0)
    cfg.setdefault("image_size",   32)

    evaluate_checkpoint(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
