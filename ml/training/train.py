"""
Phase 4 — Baseline Model Training & Validation Pipeline
------------------------------------------------------
Executes full fine-tuning of DenseNet-121 baseline on TRAIN (69,419 images)
and evaluates on VALIDATION (17,105 images) every epoch.
Official TEST split (25,596 images) remains 100% UNTOUCHED.
"""

import sys
import os
import json
import time
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.utils.seed import seed_everything
from ml.models.builder import build_model
from ml.training.loss import get_loss_function
from ml.preprocessing.loaders import get_dataloaders
from ml.evaluation.metrics import evaluate_multilabel_metrics
from ml.training.checkpointing import save_checkpoint
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def run_phase_4_training(config_path: str = "configs/model_config.yaml", data_config_path: str = "configs/data_config.yaml"):
    print("==================================================")
    print("STARTING PHASE 4 — BASELINE MODEL TRAINING & VALIDATION")
    print("==================================================")

    # 1. Load Configurations
    with open(PROJECT_ROOT / config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / data_config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # 2. Reproducibility
    seed = model_cfg.get("reproducibility", {}).get("seed", 42)
    seed_everything(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # 3. DataLoaders (TRAIN & VALIDATION ONLY - TEST IS FROZEN)
    print("\n--- Initializing PyTorch DataLoaders ---")
    train_loader, val_loader, _ = get_dataloaders(data_cfg)
    print(f"  Train DataLoader Batches: {len(train_loader)} ({len(train_loader.dataset):,} images)")
    print(f"  Validation DataLoader Batches: {len(val_loader)} ({len(val_loader.dataset):,} images)")
    print(f"  Test Set: FROZEN & UNTOUCHED")

    # 4. Model Construction
    print("\n--- Constructing DenseNet-121 Baseline ---")
    model = build_model(model_cfg).to(device)
    param_counts = model.get_parameter_counts()
    print(f"  Architecture: {model_cfg.get('model', {}).get('architecture', 'densenet121')}")
    print(f"  Total Parameters: {param_counts['total']:,}")
    print(f"  Trainable Parameters: {param_counts['trainable']:,}")

    # 5. Loss Function
    stats_path = str(PROJECT_ROOT / "data" / "processed" / "class_statistics.json")
    criterion, pos_weight_tensor = get_loss_function(model_cfg, class_stats_path=stats_path, device=device)
    print(f"\n--- Loss Function Construction ---")
    print(f"  Loss Type: Weighted BCEWithLogitsLoss")
    print(f"  Pos Weight Vector Loaded: {pos_weight_tensor.shape[0]} classes")
    print(f"  Pos Weights Sample: Infiltration={pos_weight_tensor[8].item():.4f}, Hernia={pos_weight_tensor[7].item():.4f}")

    # 6. Optimizer & Scheduler
    train_cfg = model_cfg.get("training", {})
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    epochs = int(train_cfg.get("epochs", 10))
    grad_clip = float(train_cfg.get("gradient_clipping", 1.0))
    use_amp = bool(train_cfg.get("mixed_precision", True)) and torch.cuda.is_available()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    sched_cfg = train_cfg.get("scheduler_params", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "max"),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 2)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6))
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Checkpoint Directory
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "phase4"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training History Recording
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_auroc": [],
        "val_micro_auroc": [],
        "val_macro_auprc": [],
        "val_micro_auprc": [],
        "val_macro_f1": [],
        "learning_rate": [],
        "epoch_duration_sec": [],
        "best_epoch": 0,
        "best_val_macro_auroc": 0.0
    }

    best_macro_auroc = -1.0
    start_time = time.time()

    print(f"\n==================================================")
    print(f"STARTING BASELINE TRAINING LOOP ({epochs} Epochs)")
    print(f"==================================================")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_cnt = 0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{epochs}] — Learning Rate: {current_lr:.6e}")

        for batch_idx, (images, targets, _, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item()
            batch_cnt += 1

            if batch_idx % 200 == 0 or batch_idx == len(train_loader):
                avg_b_loss = running_loss / batch_cnt
                print(f"  [Train Step {batch_idx}/{len(train_loader)}] Loss: {avg_b_loss:.4f}")

        train_epoch_loss = running_loss / batch_cnt

        # --- Validation Evaluation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)

                val_loss += loss.item()
                val_batches += 1
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        val_epoch_loss = val_loss / val_batches
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Validation Metrics
        val_metrics = evaluate_multilabel_metrics(all_logits, all_targets, PATHOLOGY_CLASSES)
        macro_auroc = val_metrics.get("macro_auroc", 0.0)
        micro_auroc = val_metrics.get("micro_auroc", 0.0)
        macro_auprc = val_metrics.get("macro_auprc", 0.0)
        micro_auprc = val_metrics.get("micro_auprc", 0.0)
        macro_f1 = val_metrics.get("macro_f1", 0.0)

        scheduler.step(macro_auroc)
        epoch_dur = time.time() - epoch_start

        # Record History
        history["epochs"].append(epoch)
        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["val_macro_auroc"].append(macro_auroc)
        history["val_micro_auroc"].append(micro_auroc)
        history["val_macro_auprc"].append(macro_auprc)
        history["val_micro_auprc"].append(micro_auprc)
        history["val_macro_f1"].append(macro_f1)
        history["learning_rate"].append(current_lr)
        history["epoch_duration_sec"].append(epoch_dur)

        print(f"\n--- Epoch [{epoch}/{epochs}] Validation Summary ---")
        print(f"  Train Loss:       {train_epoch_loss:.4f}")
        print(f"  Val Loss:         {val_epoch_loss:.4f}")
        print(f"  Val Macro AUROC:  {macro_auroc:.4f}")
        print(f"  Val Micro AUROC:  {micro_auroc:.4f}")
        print(f"  Val Macro AUPRC:  {macro_auprc:.4f}")
        print(f"  Epoch Duration:   {epoch_dur:.1f} seconds")

        # Metadata for Checkpoints
        ckpt_meta = {
            "epoch": epoch,
            "val_macro_auroc": macro_auroc,
            "val_loss": val_epoch_loss,
            "train_loss": train_epoch_loss,
            "seed": seed,
            "architecture": "densenet121",
            "pos_weights": pos_weight_tensor.cpu().tolist(),
            "class_names": PATHOLOGY_CLASSES,
            "train_manifest_sha256": "a3158bb7de313e876af199e1a4333bbcce26301b61677d8673b055501e2774b7",
            "val_manifest_sha256": "50b0eb72e7aa9322cf93afa49d4510ee211d2429083cff02bec8b173c2d6968d"
        }

        # Save Checkpoints
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metadata": ckpt_meta,
            "val_macro_auroc": macro_auroc,
            "class_names": PATHOLOGY_CLASSES,
        }

        # Save Latest Checkpoint
        save_checkpoint(state, checkpoint_dir=str(ckpt_dir), filename="latest.pth", is_best=False)

        # Save Best Checkpoint
        if macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            history["best_epoch"] = epoch
            history["best_val_macro_auroc"] = best_macro_auroc
            save_checkpoint(state, checkpoint_dir=str(ckpt_dir), filename="latest.pth", is_best=True)
            print(f"  🌟 NEW BEST VALIDATION MACRO AUROC: {best_macro_auroc:.4f}! Saved best.pth")

    total_duration = time.time() - start_time
    history["total_training_time_sec"] = total_duration

    # 7. Save Machine-Readable History JSON
    hist_json_path = PROJECT_ROOT / "data" / "processed" / "phase_4_training_history.json"
    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {hist_json_path}")

    # 8. Generate Training Curves
    generate_training_plots(history)

    # 9. Generate Training Report MD
    generate_training_report(history, total_duration, ckpt_dir)

    print("\n==================================================")
    print("PHASE 4 TRAINING COMPLETED SUCCESSFULLY")
    print("==================================================")
    return history


def generate_training_plots(history: dict):
    vis_dir = PROJECT_ROOT / "docs" / "phase_4_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    epochs = history["epochs"]

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="s", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Phase 4 DenseNet-121 — Train vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(vis_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 2. Macro AUROC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_auroc"], label="Val Macro AUROC", marker="o", color="#2ca02c")
    plt.axhline(y=max(history["val_macro_auroc"]), color="r", linestyle=":", label=f"Best ({max(history['val_macro_auroc']):.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro AUROC")
    plt.title("Phase 4 DenseNet-121 — Validation Macro AUROC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(vis_dir / "macro_auroc_curve.png", dpi=300)
    plt.close()

    # 3. Macro AUPRC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_auprc"], label="Val Macro AUPRC", marker="o", color="#9467bd")
    plt.xlabel("Epoch")
    plt.ylabel("Macro AUPRC")
    plt.title("Phase 4 DenseNet-121 — Validation Macro AUPRC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(vis_dir / "macro_auprc_curve.png", dpi=300)
    plt.close()

    # 4. Learning Rate Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["learning_rate"], label="Learning Rate", marker="^", color="#d62728")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.title("Phase 4 DenseNet-121 — Learning Rate Schedule")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(vis_dir / "lr_curve.png", dpi=300)
    plt.close()

    print(f"Saved training visualization plots to {vis_dir}/")


def generate_training_report(history: dict, total_duration: float, ckpt_dir: Path):
    report_path = PROJECT_ROOT / "docs" / "phase_4_training_report.md"
    best_ep = history["best_epoch"]
    best_auroc = history["best_val_macro_auroc"]
    last_val_loss = history["val_loss"][-1]

    content = f"""# NIH ChestX-ray14 Phase 4 — Baseline Model Training & Validation Report

**Report Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Phase 4 Training Status**: **SUCCESS**  
**Total Training Duration**: {total_duration / 60:.2f} minutes ({total_duration:.1f} seconds)  
**Best Validation Macro AUROC**: **{best_auroc:.4f}** (Epoch {best_ep})  
**Final Validation Loss**: **{last_val_loss:.4f}**  

---

## 📊 1. Dataset & Patient Split Information

- **Dataset**: NIH ChestX-ray14
- **Total Dataset Size**: 112,120 PNG images ($1024 \\times 1024$ raw, resized to $320 \\times 320$)
- **Train Split**: 69,419 images (22,406 patients)
- **Validation Split**: 17,105 images (5,602 patients)
- **Test Split**: 25,596 images (2,797 patients) — **100% UNTOUCHED & FROZEN**
- **Patient Overlap**: 0 across all splits (Strict patient-disjoint governance)

---

## ⚙️ 2. Baseline Model Architecture & Hyperparameters

- **Architecture**: DenseNet-121 (`torchvision.models.densenet121`)
- **Pretrained Weights**: `DenseNet121_Weights.DEFAULT` (ImageNet-1K V1)
- **Classification Head**: `nn.Linear(in_features=1024, out_features=14)`
- **Total Parameters**: 6,968,206
- **Trainable Parameters**: 6,968,206 (Full Fine-tuning)
- **Loss Function**: `nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)`
- **Class Weights**: Loaded dynamically from `data/processed/class_statistics.json`
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-2`)
- **Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=2)`
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`)
- **Seed**: `42` (Global PyTorch, NumPy, and CUDA seed)

---

## 📈 3. Per-Epoch Training & Validation Progress Table

| Epoch | Train Loss | Val Loss | Val Macro AUROC | Val Micro AUROC | Val Macro AUPRC | Learning Rate | Duration (s) | Checkpoint Status |
|---|---|---|---|---|---|---|---|---|
"""
    for idx in range(len(history["epochs"])):
        ep = history["epochs"][idx]
        tr_l = history["train_loss"][idx]
        vl_l = history["val_loss"][idx]
        m_auroc = history["val_macro_auroc"][idx]
        mi_auroc = history["val_micro_auroc"][idx]
        m_auprc = history["val_macro_auprc"][idx]
        lr_val = history["learning_rate"][idx]
        dur = history["epoch_duration_sec"][idx]
        status = "🌟 BEST" if ep == best_ep else "Saved"
        content += f"| {ep} | {tr_l:.4f} | {vl_l:.4f} | **{m_auroc:.4f}** | {mi_auroc:.4f} | {m_auprc:.4f} | {lr_val:.2e} | {dur:.1f}s | {status} |\n"

    content += f"""
---

## 📁 4. Deliverables & Saved Checkpoints

- **Best Checkpoint**: `{ckpt_dir}/best.pth`
- **Latest Checkpoint**: `{ckpt_dir}/latest.pth`
- **Machine-Readable History**: `data/processed/phase_4_training_history.json`
- **Training Curves**: `docs/phase_4_visualizations/`
- **Verification Script**: `scripts/verify_phase_4.py`

---

## ⚠️ 5. Medical Research & Safety Disclaimer

> [!IMPORTANT]
> This DenseNet-121 baseline model is strictly an experimental multi-label research baseline trained on the NIH ChestX-ray14 dataset. It is **NOT** a clinically certified diagnostic device and must never be used for primary patient diagnosis or clinical decision-making.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved Phase 4 training report to {report_path}")


if __name__ == "__main__":
    run_phase_4_training()
