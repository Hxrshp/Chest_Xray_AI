"""
ChestXrayTrainer — Multi-Label Training & Validation Engine
------------------------------------------------------------
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ml.utils.seed import seed_everything

from ml.evaluation.metrics import evaluate_multilabel_metrics
from ml.training.checkpointing import save_checkpoint


class ChestXrayTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training hyperparams
        training_cfg = self.config.get("training", {})
        self.grad_clip = float(training_cfg.get("gradient_clipping", 1.0))
        self.use_amp = bool(training_cfg.get("mixed_precision", True)) and self.device.type == "cuda"
        self.grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Metrics & Checkpointing
        self.best_metric = -1.0
        self.best_epoch = 0

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, max_batches: Optional[int] = None) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        self.optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            batch_size = images.size(0)
            running_loss += (loss.item() * self.grad_accum_steps) * batch_size
            total_samples += batch_size

        return running_loss / max(total_samples, 1)

    @torch.no_grad()
    def validate_epoch(
        self, val_loader: torch.utils.data.DataLoader, max_batches: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_logits = []
        all_targets = []

        for batch_idx, (images, targets) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        val_loss = running_loss / max(total_samples, 1)

        y_logits = torch.cat(all_logits, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()

        metrics = evaluate_multilabel_metrics(y_true, y_logits)
        metrics["val_loss"] = val_loss
        return val_loss, metrics

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 1,
        checkpoint_dir: Optional[str] = None,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "val_loss": [], "macro_auroc": [], "macro_auprc": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, max_batches=max_train_batches)
            val_loss, metrics = self.validate_epoch(val_loader, max_batches=max_val_batches)
            elapsed = time.time() - t0

            macro_auroc = metrics.get("macro_auroc", 0.0)
            macro_auprc = metrics.get("macro_auprc", 0.0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["macro_auroc"].append(macro_auroc)
            history["macro_auprc"].append(macro_auprc)

            # Learning Rate Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(macro_auroc)
                else:
                    self.scheduler.step()

            # Checkpoint saving
            is_best = macro_auroc > self.best_metric
            if is_best:
                self.best_metric = macro_auroc
                self.best_epoch = epoch

            if checkpoint_dir:
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "best_metric": self.best_metric,
                    "metrics": metrics,
                    "config": self.config,
                }
                save_checkpoint(state, checkpoint_dir, filename="latest.pth", is_best=is_best)

            print(
                f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro AUROC: {macro_auroc:.4f} | Val Macro AUPRC: {macro_auprc:.4f}"
            )

        return history
