import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def _gpu_mem_gb(kind: str = "allocated") -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    if kind == "allocated":
        b = torch.cuda.max_memory_allocated()
    elif kind == "reserved":
        b = torch.cuda.max_memory_reserved()
    else:
        b = torch.cuda.memory_allocated()
    return b / (1024**3)


class MemorySpeedMonitor(pl.Callback):
    """
    Track VRAM, throughput, and CPU RAM.
    Writes JSONL under <run_dir>/<out_dir>/eff_<tag>.jsonl
    Only rank 0 writes.
    """
    def __init__(self, out_dir: str, tag: str = "train", log_every_n_steps: int = 50, rank_zero_only: bool = True):
        super().__init__()
        self.out_dir_rel = Path(out_dir)                 # relative; resolved at fit start
        self.tag = tag
        self.log_every_n_steps = int(log_every_n_steps)
        self.rank_zero_only = rank_zero_only

        self.run_dir: Optional[Path] = None              # set at fit start
        self.out_dir_abs: Optional[Path] = None
        self.jsonl: Optional[Path] = None
        self.reset_state()

    def reset_state(self):
        self.step0_time = None
        self.epoch_start_time = None
        self.step_count = 0
        self.items_count = 0

    # ---------- helpers ----------

    def _is_rank0(self, trainer) -> bool:
        if not self.rank_zero_only:
            return True
        # Works for single GPU and DDP
        try:
            return getattr(trainer, "global_rank", 0) == 0
        except Exception:
            return True

    def _ensure_paths(self, trainer):
        # Resolve run directory once training is about to start
        if self.run_dir is None:
            # Prefer logger's log_dir, fall back to default_root_dir, then CWD
            log_dir = None
            try:
                # pl >= 2: logger has .log_dir on CSV/W&B; else use .save_dir/.version
                if trainer.logger is not None:
                    log_dir = getattr(trainer.logger, "log_dir", None)
                    if log_dir is None and hasattr(trainer.logger, "save_dir"):
                        # CSVLogger style
                        save_dir = Path(trainer.logger.save_dir)
                        version = getattr(trainer.logger, "version", None)
                        log_dir = str(save_dir / str(version)) if version is not None else str(save_dir)
            except Exception:
                log_dir = None

            base = Path(log_dir) if log_dir else Path(getattr(trainer, "default_root_dir", "."))  # hydra run dir
            self.run_dir = base.resolve()
            self.out_dir_abs = (self.run_dir / self.out_dir_rel).resolve()
            self.out_dir_abs.mkdir(parents=True, exist_ok=True)
            self.jsonl = self.out_dir_abs / f"eff_{self.tag}.jsonl"

        # Make sure dir still exists before each write
        if self.out_dir_abs is not None:
            self.out_dir_abs.mkdir(parents=True, exist_ok=True)

    # ---------- hooks ----------

    def on_fit_start(self, trainer, pl_module):
        if not self._is_rank0(trainer):
            return
        self._ensure_paths(trainer)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_start(self, trainer, pl_module):
        if self.tag != "train" or not self._is_rank0(trainer):
            return
        self._ensure_paths(trainer)
        self.epoch_start_time = time.perf_counter()
        self.step0_time = None
        self.step_count = 0
        self.items_count = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.tag != "train" or not self._is_rank0(trainer):
            return
        if self.step0_time is None:
            self.step0_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.tag != "train" or not self._is_rank0(trainer):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_count += 1

        # try to infer "batch size" / items in this batch
        bs = getattr(batch, "batch_size", None)
        if bs is None:
            # PyG Batch: len(batch) -> num graphs, or fallback 0
            try:
                bs = len(batch)
            except Exception:
                bs = 0
        self.items_count += int(bs)

        if self.step_count % self.log_every_n_steps == 0 and self.jsonl is not None:
            now = time.perf_counter()
            dt = now - (self.step0_time or now)
            steps_per_s = self.log_every_n_steps / dt if dt > 0 else 0.0
            it_per_s = self.items_count / (now - (self.epoch_start_time or now)) if self.epoch_start_time else 0.0
            rec = {
                "phase": "train",
                "global_step": int(trainer.global_step),
                "steps_per_s": steps_per_s,
                "items_seen": int(self.items_count),
                "items_per_s_epoch_avg": it_per_s,
                "gpu_peak_allocated_gb": _gpu_mem_gb("allocated"),
                "gpu_peak_reserved_gb": _gpu_mem_gb("reserved"),
                "cpu_rss_gb": psutil.Process(os.getpid()).memory_info().rss / (1024**3) if _HAS_PSUTIL else None,
                "time": time.time(),
            }
            # make sure directory exists and append
            self._ensure_paths(trainer)
            with open(self.jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[EFF][train] step={rec['global_step']} "
                  f"steps/s={rec['steps_per_s']:.2f} items/s~{rec['items_per_s_epoch_avg']:.1f} "
                  f"gpu_peak={rec['gpu_peak_allocated_gb']:.2f}GB "
                  f"reserved={rec['gpu_peak_reserved_gb']:.2f}GB "
                  f"cpu={rec['cpu_rss_gb'] if rec['cpu_rss_gb'] is not None else 'NA'}GB")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.tag != "train" or not self._is_rank0(trainer):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total = time.perf_counter() - (self.epoch_start_time or time.perf_counter())
        rec = {
            "phase": "train_epoch_end",
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "epoch_seconds": total,
            "items_seen": int(self.items_count),
            "items_per_s": (self.items_count / total) if total > 0 else 0.0,
            "gpu_peak_allocated_gb": _gpu_mem_gb("allocated"),
            "gpu_peak_reserved_gb": _gpu_mem_gb("reserved"),
            "cpu_rss_gb": psutil.Process(os.getpid()).memory_info().rss / (1024**3) if _HAS_PSUTIL else None,
            "time": time.time(),
        }
        self._ensure_paths(trainer)
        if self.jsonl is not None:
            with open(self.jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"[EFF][epoch] {rec}")