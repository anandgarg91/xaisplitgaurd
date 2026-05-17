"""
experiments/run_experiment.py
──────────────────────────────
Main training script for XAI-SplitShield experiments.

Supports:
  - Single-client and multi-client split learning
  - All four backdoor attacks (BadNets, Blended, WaNet, LIRA)
  - XAI-SplitShield defense (SDA + AAS + AGNS + AWSA)
  - Baseline defenses (SplitGuard, STRIP-SL, no defense)
  - Ablation modes

Usage:
    python experiments/run_experiment.py \\
        --dataset cifar10 --attack badnets --defense xai_splitshield \\
        --split_layer 2 --poison_rate 0.1 --epochs 100

    python experiments/run_experiment.py --config configs/cifar10_badnets.yaml
"""

from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from data.dataset_loader import get_dataset
from data.poisoning import PoisonedDataLoader, get_attack
from models.resnet import build_split_resnet18, get_smashed_shape
from defense.xai_splitshield import XAISplitShield
from utils.metrics import (
    compute_asr, compute_clean_accuracy, MetricsTracker
)
from utils.logger import ExperimentLogger
from utils.visualization import (
    plot_aas_timeseries, plot_training_curves, plot_asr_ca_comparison
)


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(args: argparse.Namespace) -> dict:
    """Merges YAML config with CLI overrides (CLI takes priority)."""
    cfg: dict = {}

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # Flatten nested YAML into one-level dict for convenience
    flat: dict = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values

    # CLI overrides
    overrides = {
        "dataset":      args.dataset,
        "attack":       args.attack,
        "defense":      args.defense,
        "split_layer":  args.split_layer,
        "poison_rate":  args.poison_rate,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "seed":         args.seed,
        "num_clients":  args.num_clients,
        "num_malicious":args.num_malicious,
        "target_class": args.target_class,
        "log_dir":      args.log_dir,
        "adaptive":     args.adaptive_attack,
        "ablation":     args.ablation,
    }
    for k, v in overrides.items():
        if v is not None:
            flat[k] = v

    # Defaults
    flat.setdefault("dataset",       "cifar10")
    flat.setdefault("attack",        "badnets")
    flat.setdefault("defense",       "xai_splitshield")
    flat.setdefault("split_layer",   2)
    flat.setdefault("poison_rate",   0.1)
    flat.setdefault("epochs",        100)
    flat.setdefault("batch_size",    128)
    flat.setdefault("lr",            0.01)
    flat.setdefault("seed",          42)
    flat.setdefault("num_clients",   1)
    flat.setdefault("num_malicious", 0)
    flat.setdefault("target_class",  0)
    flat.setdefault("log_dir",       "./results")
    flat.setdefault("warmup_epochs", 5)
    flat.setdefault("image_size",    32)
    flat.setdefault("num_workers",   4)
    flat.setdefault("weight_decay",  5e-4)
    flat.setdefault("adaptive",      False)
    flat.setdefault("ablation",      False)

    return flat


# ── Single-Round Training Step ────────────────────────────────────────────────

def train_step(
    images, labels, is_poisoned_gt,
    client_model, server_model,
    client_optimizer, server_optimizer,
    criterion, device,
    shield: XAISplitShield | None,
    warmup: bool,
    cfg: dict,
) -> dict:
    """One training batch — full SL forward/backward + optional defense."""
    images = images.to(device)
    labels = labels.to(device)

    client_optimizer.zero_grad()
    server_optimizer.zero_grad()

    # ── Client forward ──────────────────────────────────────────────────────
    z = client_model(images)
    z_leaf = z.detach().requires_grad_(True)   # Cut gradient at split

    # ── Defense (warm-up or online) ─────────────────────────────────────────
    aas_score   = 0.0
    is_poisoned = False
    z_defended  = z_leaf

    if shield is not None:
        if warmup:
            shield.warmup(z_leaf.detach(), labels)
        else:
            z_defended, report = shield.defend(z_leaf.detach(), labels)
            z_defended = z_defended.detach().requires_grad_(True)
            aas_score   = report.aas_score
            is_poisoned = report.is_poisoned

    # ── Server forward + backward (explicitly grad-enabled) ────────────────
    # We wrap with torch.enable_grad() to defend against any defense module
    # that may have left a torch.no_grad() context or eval-mode state behind.
    # Also ensure both models are in train mode for this step.
    client_model.train()
    server_model.train()

    with torch.enable_grad():
        logits = server_model(z_defended)
        loss   = criterion(logits, labels)

        if not loss.requires_grad:
            # Fail loudly with a useful message instead of the generic
            # "element 0 of tensors does not require grad" deep in autograd.
            raise RuntimeError(
                "loss.requires_grad=False — the server forward pass produced "
                "a non-grad-tracking tensor. This usually means the defense "
                "module left an autograd context inconsistent. Check that "
                "every defense method restores model.train() after eval(), "
                "and that no torch.no_grad() block escapes."
            )

        # ── Server backward ────────────────────────────────────────────────
        loss.backward()
        server_optimizer.step()

        # ── Client backward (via smashed-data gradient) ────────────────────
        grad_z = z_defended.grad if z_defended.requires_grad else z_leaf.grad
        if grad_z is not None:
            z.backward(grad_z)
            client_optimizer.step()

    with torch.no_grad():
        preds   = logits.argmax(dim=-1)
        acc     = (preds == labels).float().mean().item()

    return {
        "loss":       loss.item(),
        "acc":        acc,
        "aas_score":  aas_score,
        "is_poisoned":is_poisoned,
        "is_gt_poisoned": bool(is_poisoned_gt.any().item()),
    }


# ── Main Training Loop ────────────────────────────────────────────────────────

def run(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = (f"{cfg['dataset']}_{cfg['attack']}_{cfg['defense']}"
                f"_split{cfg['split_layer']}_s{cfg['seed']}")
    exp_log = ExperimentLogger(run_name, cfg["log_dir"], cfg.get("tensorboard", True), cfg)
    exp_log.info(f"Running: {run_name}")
    exp_log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader, num_classes = get_dataset(
        name=cfg["dataset"],
        root=cfg.get("root", "./data/raw"),
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    attack = get_attack(
        name=cfg["attack"],
        cfg=cfg,
        image_size=cfg["image_size"],
        num_classes=num_classes,
        device=device,
    )
    poisoned_loader = PoisonedDataLoader(
        train_loader, attack,
        poison_rate=cfg["poison_rate"],
        adaptive=cfg.get("adaptive", False),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    client_model, server_model = build_split_resnet18(
        num_classes=num_classes,
        split_layer=cfg["split_layer"],
        pretrained=cfg.get("pretrained", False),
    )
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    smashed_shape = get_smashed_shape(client_model, cfg["image_size"], device=device)
    exp_log.info(f"Smashed data shape: {smashed_shape}  (d={int(np.prod(smashed_shape))})")

    # ── Optimizers ────────────────────────────────────────────────────────────
    client_optimizer = optim.SGD(
        client_model.parameters(), lr=cfg["lr"],
        momentum=0.9, weight_decay=cfg["weight_decay"]
    )
    server_optimizer = optim.SGD(
        server_model.parameters(), lr=cfg["lr"],
        momentum=0.9, weight_decay=cfg["weight_decay"]
    )
    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(client_optimizer, T_max=cfg["epochs"])
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    # ── Defense ───────────────────────────────────────────────────────────────
    shield = None
    if cfg["defense"] == "xai_splitshield":
        shield = XAISplitShield(
            server_model=server_model,
            smashed_shape=tuple(smashed_shape),
            device=device,
            cfg=cfg,
            num_clients=cfg["num_clients"],
        )

    # ── Training ──────────────────────────────────────────────────────────────
    train_accs, val_accs, train_losses = [], [], []
    all_aas_scores, all_poisoned_gt = [], []

    for epoch in range(1, cfg["epochs"] + 1):
        warmup_phase = (epoch <= cfg["warmup_epochs"])

        if warmup_phase:
            exp_log.info(f"[Epoch {epoch}] WARM-UP PHASE")
        elif epoch == cfg["warmup_epochs"] + 1 and shield is not None:
            shield.calibrate()
            exp_log.info(f"Defense calibrated. Threshold τ = {shield.aas_detector.threshold:.4f}")

        client_model.train()
        server_model.train()
        tracker = MetricsTracker()

        for images, labels, is_poisoned_gt in poisoned_loader:
            step = train_step(
                images, labels, is_poisoned_gt,
                client_model, server_model,
                client_optimizer, server_optimizer,
                criterion, device,
                shield, warmup_phase, cfg,
            )
            tracker.update(
                loss=step["loss"],
                acc=step["acc"],
                aas=step["aas_score"],
                is_poisoned=step["is_poisoned"],
                is_gt_poisoned=step["is_gt_poisoned"],
            )
            all_aas_scores.append(step["aas_score"])
            all_poisoned_gt.append(step["is_gt_poisoned"])

        scheduler_c.step()
        scheduler_s.step()

        summary = tracker.summary()
        train_accs.append(summary["acc"])
        train_losses.append(summary["loss"])

        # Epoch-level evaluation
        ca = compute_clean_accuracy(client_model, server_model, test_loader,
                                    device, shield if not warmup_phase else None)
        val_accs.append(ca)

        asr = 0.0
        if attack is not None and not warmup_phase:
            asr = compute_asr(client_model, server_model, test_loader,
                              attack, cfg["target_class"], device,
                              shield if not warmup_phase else None)

        exp_log.log_epoch(epoch, {
            "loss": summary["loss"], "train_acc": summary["acc"],
            "val_acc": ca, "asr": asr, "aas_mean": summary["aas_mean"],
            "precision": summary["precision"], "recall": summary["recall"],
            "f1": summary["f1"],
        })

    # ── Final Evaluation ──────────────────────────────────────────────────────
    final_ca  = compute_clean_accuracy(client_model, server_model, test_loader, device, shield)
    final_asr = 0.0
    if attack is not None:
        final_asr = compute_asr(client_model, server_model, test_loader,
                                attack, cfg["target_class"], device, shield)

    exp_log.log_final({
        "run_name": run_name,
        "final_ca":  round(final_ca * 100, 2),
        "final_asr": round(final_asr * 100, 2),
        "config":    cfg,
    })
    exp_log.info(f"\n{'='*50}")
    exp_log.info(f"FINAL  CA = {final_ca*100:.2f}%   ASR = {final_asr*100:.2f}%")
    exp_log.info(f"{'='*50}\n")

    # ── Visualizations ────────────────────────────────────────────────────────
    vis_dir = Path(cfg["log_dir"]) / run_name / "plots"
    vis_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(train_accs, val_accs, train_losses,
                         title=run_name,
                         save_path=str(vis_dir / "training_curves.png"))

    if shield is not None and shield._warmup_done:
        plot_aas_timeseries(
            all_aas_scores, all_poisoned_gt,
            threshold=shield.aas_detector.threshold,
            save_path=str(vis_dir / "aas_timeseries.png"),
        )

    # Save models
    ckpt_dir = Path(cfg["log_dir"]) / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(client_model.state_dict(), ckpt_dir / "client_model.pth")
    torch.save(server_model.state_dict(), ckpt_dir / "server_model.pth")
    exp_log.info(f"Models saved to {ckpt_dir}")
    exp_log.close()

    return {"ca": final_ca, "asr": final_asr}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAI-SplitShield Experiment Runner")
    p.add_argument("--config",          type=str,   default=None)
    p.add_argument("--dataset",         type=str,   default=None,
                   choices=["cifar10", "gtsrb", "xray"])
    p.add_argument("--attack",          type=str,   default=None,
                   choices=["badnets", "blended", "wanet", "lira", "none"])
    p.add_argument("--defense",         type=str,   default=None,
                   choices=["xai_splitshield", "none"])
    p.add_argument("--split_layer",     type=int,   default=None)
    p.add_argument("--poison_rate",     type=float, default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--num_clients",     type=int,   default=None)
    p.add_argument("--num_malicious",   type=int,   default=None)
    p.add_argument("--target_class",    type=int,   default=None)
    p.add_argument("--log_dir",         type=str,   default=None)
    p.add_argument("--adaptive_attack", action="store_true", default=None)
    p.add_argument("--ablation",        action="store_true", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
