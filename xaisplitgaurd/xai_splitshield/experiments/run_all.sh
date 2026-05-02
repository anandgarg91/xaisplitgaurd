#!/usr/bin/env bash
# ============================================================
#  run_all.sh — Reproduce all XAI-SplitShield paper experiments
#  Runs Table 2 (main results), ablation, and adaptive adversary
#
#  Usage:
#    bash experiments/run_all.sh
#    bash experiments/run_all.sh --gpu 0 --epochs 50   # quick run
# ============================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────
EPOCHS=100
GPU=0
LOG_DIR="./results"
SEEDS="42 123 456 789 2024"    # 5 seeds → mean ± std

# ── Parse CLI flags ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --gpu)    GPU="$2";    shift 2 ;;
    --seeds)  SEEDS="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES=$GPU
RUNNER="python experiments/run_experiment.py"

echo "============================================================"
echo "  XAI-SplitShield — Full Experiment Suite"
echo "  Epochs=$EPOCHS  GPU=$GPU  Log=$LOG_DIR"
echo "============================================================"

# ── Helper: run one config across all seeds ───────────────────
run_seeds() {
  local DATASET=$1 ATTACK=$2 DEFENSE=$3 EXTRA="${4:-}"
  for SEED in $SEEDS; do
    echo ""
    echo ">>> [$DATASET | $ATTACK | $DEFENSE | seed=$SEED]"
    $RUNNER \
      --dataset "$DATASET" \
      --attack  "$ATTACK"  \
      --defense "$DEFENSE" \
      --epochs  "$EPOCHS"  \
      --seed    "$SEED"    \
      --log_dir "$LOG_DIR" \
      $EXTRA
  done
}

# ──────────────────────────────────────────────────────────────
# TABLE 2: Main Results (3 datasets × 4 attacks × 2 defenses)
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== TABLE 2: CIFAR-10 ==="
for ATK in badnets blended wanet lira; do
  run_seeds cifar10 "$ATK" xai_splitshield
  run_seeds cifar10 "$ATK" none              # Undefended baseline
done

echo ""
echo "=== TABLE 2: GTSRB ==="
for ATK in badnets blended wanet lira; do
  run_seeds gtsrb "$ATK" xai_splitshield
  run_seeds gtsrb "$ATK" none
done

echo ""
echo "=== TABLE 2: X-Ray ==="
for ATK in badnets blended wanet lira; do
  run_seeds xray "$ATK" xai_splitshield
  run_seeds xray "$ATK" none
done

# ──────────────────────────────────────────────────────────────
# ABLATION (Table 3): CIFAR-10, WaNet — component ablation
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== ABLATION: CIFAR-10 + WaNet ==="
for SEED in $SEEDS; do
  # Full system (baseline for ablation)
  $RUNNER --dataset cifar10 --attack wanet --defense xai_splitshield \
          --epochs "$EPOCHS" --seed "$SEED" --log_dir "$LOG_DIR"

  # w/o SHAP component (alpha=0, LRP+LIME only) — set sda_alpha=0 via env
  SDA_ALPHA=0.0 $RUNNER --dataset cifar10 --attack wanet --defense xai_splitshield \
          --epochs "$EPOCHS" --seed "$SEED" --log_dir "${LOG_DIR}/ablation_no_shap"

  # w/o AGNS (suppress disabled via epsilon_acc=100%)
  AGNS_EPSILON=1.0 $RUNNER --dataset cifar10 --attack wanet --defense xai_splitshield \
          --epochs "$EPOCHS" --seed "$SEED" --log_dir "${LOG_DIR}/ablation_no_agns"
done

# ──────────────────────────────────────────────────────────────
# ADAPTIVE ADVERSARY: CIFAR-10, all attacks
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== ADAPTIVE ADVERSARY: CIFAR-10 ==="
for ATK in badnets blended wanet lira; do
  run_seeds cifar10 "$ATK" xai_splitshield "--adaptive_attack"
done

# ──────────────────────────────────────────────────────────────
# MULTI-CLIENT AWSA: CIFAR-10, WaNet, 10 clients / 3 malicious
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== MULTI-CLIENT AWSA: CIFAR-10 + WaNet ==="
for SEED in $SEEDS; do
  $RUNNER --dataset cifar10 --attack wanet --defense xai_splitshield \
          --num_clients 10 --num_malicious 3 \
          --epochs "$EPOCHS" --seed "$SEED" --log_dir "${LOG_DIR}/multi_client"
done

echo ""
echo "============================================================"
echo "  All experiments complete. Results in: $LOG_DIR"
echo "  Run: python experiments/collect_results.py --log_dir $LOG_DIR"
echo "============================================================"
