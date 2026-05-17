# Generating Real Plots from Experiment Logs

The script `scripts/plot_real_results.py` reads the result tree your experiments produce and generates publication-quality figures from **actual measured values** — no synthetic data, no fabricated numbers.

## What the script reads

For every experiment, `run_experiment.py` creates a result directory at `results/<run_name>/` containing:

| File              | Contents                                                          |
|-------------------|-------------------------------------------------------------------|
| `config.json`     | Full hyperparameter snapshot (dataset, attack, defense, etc.)     |
| `metrics.jsonl`   | One JSON record per epoch — see schema below                      |
| `summary.json`    | Final clean accuracy (`final_ca`) and attack success rate (`final_asr`) |
| `plots/`          | The auto-generated training-curves PNG written by `run_experiment.py` itself |
| `tb/`             | TensorBoard event files (optional)                                |

Each line in `metrics.jsonl` follows this schema:

```json
{
  "epoch": 1,
  "phase": "train",
  "loss": 1.654,
  "train_acc": 0.341,
  "val_acc": 0.323,
  "asr": 0.008,
  "aas_mean": 0.272,
  "precision": 0.625,
  "recall": 0.626,
  "f1": 0.625
}
```

Run names follow the convention emitted by `run_experiment.py`:

```
<dataset>_<attack>_<defense>_split<N>_s<seed>
```

For example: `cifar10_badnets_xai_splitshield_split2_s42`

## How to run

After populating `./results/` with one or more experiments:

```bash
python scripts/plot_real_results.py --log_dir ./results
```

Or with a custom output directory:

```bash
python scripts/plot_real_results.py --log_dir ./results --out_dir ./paper_figs
```

Or filtered to a single dataset:

```bash
python scripts/plot_real_results.py --log_dir ./results --dataset cifar10
```

## What it produces

The script writes the following files to `<out_dir>/`:

| File                        | What it shows                                                          | Needs                                            |
|-----------------------------|------------------------------------------------------------------------|--------------------------------------------------|
| `fig_training_curves.png`   | Train/val accuracy + loss over epochs, one panel per condition         | metrics.jsonl with `loss`, `train_acc`, `val_acc` |
| `fig_aas_evolution.png`     | AAS metric over rounds with detection threshold τ (G2 visualization)   | metrics.jsonl with `aas_mean`, defense=xai_splitshield |
| `fig_detection_metrics.png` | Precision / recall / F1 of AAS detection over epochs (G2)              | metrics.jsonl with `precision`, `recall`, `f1`   |
| `fig_asr_vs_byzantine.png`  | Final CA and ASR vs malicious-client fraction (G3 — Theorem 2)         | Multiple runs with different `num_malicious`     |
| `fig_defense_comparison.png`| Bar chart: CA and ASR per defense across datasets                      | Multiple defenses (`none` vs `xai_splitshield`)  |
| `fig_attack_comparison.png` | ASR for each (dataset, attack) pair with vs without defense            | Runs with `--defense none` and `--defense xai_splitshield` |
| `summary_table.csv`         | Mean ± std for CA and ASR per (dataset, attack, defense, f/N) condition| Any completed run                                |

Each plot automatically:
- **Averages across seeds** if you ran multiple `--seed` values for the same condition, drawing mean lines with shaded ±1σ bands
- **Skips itself with a clear message** if the data it needs isn't present (no spurious empty plots)
- **Handles partial data** — a single seed produces a single line instead of a band

## Recommended experiment sets

To populate the figures for each of the three formal guarantees:

### G1 (Shapley axioms)
G1 figures (axiom verification) require per-feature attribution values which the codebase doesn't currently log. You would need to add a `log_axiom_check` call in the SDA module — see the `Limitations` section below.

### G2 (AAS consistency)
Run the main attack-defense matrix:

```bash
for atk in badnets blended wanet; do
  for dfn in none xai_splitshield; do
    for s in 42 43 44 45 46; do
      python experiments/run_experiment.py \
        --dataset cifar10 --attack $atk --defense $dfn \
        --split_layer 2 --seed $s --epochs 100
    done
  done
done
```

This produces `fig_aas_evolution.png`, `fig_detection_metrics.png`, `fig_defense_comparison.png`, and `fig_attack_comparison.png`.

### G3 (AWSA Byzantine convergence)
Sweep `f/N`:

```bash
for fm in 0 1 2 3; do        # 0/10, 1/10, 2/10, 3/10
  for s in 42 43 44; do
    python experiments/run_experiment.py \
      --dataset cifar10 --attack badnets --defense xai_splitshield \
      --num_clients 10 --num_malicious $fm --seed $s --epochs 100
  done
done
```

This produces `fig_asr_vs_byzantine.png`.

## Limitations

Some of the synthetic figures from the earlier "formal guarantees" plot generator cannot be reproduced from the current logs:

| Figure                  | Why it can't come from current logs                              | What to log                                       |
|-------------------------|------------------------------------------------------------------|---------------------------------------------------|
| Shapley efficiency      | Requires per-batch `Σ φ_j` and `f^s_c(Z) - f^s_c(Z_0)`           | Add `log_axiom_check(phi, logits, baseline)` in SDA |
| Symmetry/dummy/additivity | Needs per-feature φ values + ground-truth feature semantics  | Synthetic-feature unit test in `tests/`           |
| AAS log-log scaling     | Needs AAS computed for *varying batch sizes* in one run          | Add a `--batch_size_sweep` mode to evaluation     |
| ROC curve               | Needs raw AAS scores for individual batches, both clean & poisoned | Log per-batch AAS to a separate JSONL during eval |
| AWSA trust-weight history | Needs per-client `w_i(t)` per round                            | Add `self.awsa.log_weights()` after each aggregation |

For the paper's main empirical claims (G2 detection + G3 convergence), the current logging schema is sufficient and the script above produces what you need. The other figures are nice-to-have axiom verifications that would strengthen the appendix but aren't strictly necessary for the headline results.

## Adding new metrics

To extend logging with extra fields, edit the `exp_log.log_epoch(...)` call in `experiments/run_experiment.py:328` to include them. The plot generator will pick up any column it knows about automatically — for new columns, add a new `plot_xxx(runs, out_dir)` function in `plot_real_results.py` following the existing pattern.
