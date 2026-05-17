"""
utils/logger.py
───────────────
Experiment logging: console + file + optional TensorBoard.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_logger(name: str, log_dir: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger with console + optional file handler.

    Args:
        name:    Logger name (typically __name__ of calling module).
        log_dir: Directory to write log file. None = console only.
        level:   Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            Path(log_dir) / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class ExperimentLogger:
    """
    Structured experiment logger that records per-epoch metrics to:
      - Console (via Python logging)
      - JSON lines file (results/<run_name>/metrics.jsonl)
      - TensorBoard (optional, if tensorboard is installed)

    Args:
        run_name:   Unique name for this run (e.g. "cifar10_badnets_xai").
        log_dir:    Root directory for all run logs.
        use_tb:     Enable TensorBoard writer.
        cfg:        Full config dict saved alongside metrics.
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str = "./results",
        use_tb: bool = True,
        cfg: Optional[Dict] = None,
    ):
        self.run_name = run_name
        self.run_dir  = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(run_name, str(self.run_dir))

        # JSON lines output
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._jsonl_file = open(self.jsonl_path, "a")

        # Save config snapshot
        if cfg is not None:
            with open(self.run_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2)

        # TensorBoard
        self.tb_writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
                self.logger.info("TensorBoard writer initialized.")
            except ImportError:
                self.logger.warning("TensorBoard not available; skipping.")

    # ── Logging API ───────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, metrics: Dict[str, Any], phase: str = "train"):
        """
        Logs one epoch's metrics dict.

        Args:
            epoch:   Current epoch number.
            metrics: Dict of metric name → value.
            phase:   "train" | "val" | "test".
        """
        record = {"epoch": epoch, "phase": phase, **metrics}
        self.logger.info(
            f"[Epoch {epoch:03d}][{phase.upper()}] " +
            " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in metrics.items())
        )
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{phase}/{k}", v, epoch)

    def log_final(self, results: Dict[str, Any]):
        """Writes final experiment summary to JSON."""
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Final summary saved to {summary_path}")
        self.logger.info("Results: " + str(results))

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def close(self):
        self._jsonl_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
