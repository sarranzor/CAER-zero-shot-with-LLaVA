#### Utils for Double ResNet + CCIM ablation studies 

### Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix
)

from dual_resnet_shared import (
    device,
    FEAT_CTX_DIM, FEAT_BODY_DIM,
    LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, GRAD_CLIP,
    ContextEncoder, BodyEncoder,
)

### ContextEncoder + CCIM
class ContextOnlyModel(nn.Module):
    def __init__(self, ctx_dim: int = FEAT_CTX_DIM, n_classes: int = 7,
                 freeze_backbone: bool = False):
        super().__init__()
        self.encoder = ContextEncoder(ctx_dim, freeze_backbone)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(ctx_dim, n_classes),
        )

    def forward(self, ctx_img, body_img=None):
        return self.head(self.encoder(ctx_img))

### BodyEncoder + CCIM
class BodyOnlyModel(nn.Module):
    def __init__(self, body_dim: int = FEAT_BODY_DIM, n_classes: int = 7,
                 freeze_backbone: bool = False):
        super().__init__()
        self.encoder = BodyEncoder(body_dim, freeze_backbone)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(body_dim, n_classes),
        )

    def forward(self, ctx_img, body_img):
        return self.head(self.encoder(body_img))

### Training
def train_ablation_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    tag,
    ckpt_path,
    max_epochs = MAX_EPOCHS,
    patience = PATIENCE,
):
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, no_improve = 0.0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for ctx, body, labels in tqdm(train_loader, desc=f"[{tag}] Ep {epoch}/{max_epochs}", leave=False):
            ctx, body, labels = ctx.to(device), body.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ctx, body), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for ctx, body, labels in val_loader:
                ctx, body, labels = ctx.to(device), body.to(device), labels.to(device)
                logits = model(ctx, body)
                val_loss += criterion(logits, labels).item()
                preds.extend(logits.argmax(1).cpu().tolist())
                trues.extend(labels.cpu().tolist())

        avg_val = val_loss / len(val_loader)
        val_acc = balanced_accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        scheduler.step()

        print(f"Epoch {epoch:3d} | train={avg_train:.4f} | val={avg_val:.4f} "
              f"| mean-acc={val_acc:.4f} | F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f" NEW BEST (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    history["best_val_f1"] = best_f1
    return history

### Evaluation
def evaluate_ablation_model(
    model,
    loader,
    label_names,
    tag,
    save_prefix,
    dataset_key = None,
):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ctx, body, labels in tqdm(loader, desc=f"Evaluating {tag}"):
            ctx, body = ctx.to(device), body.to(device)
            if dataset_key is not None:
                logits = model(ctx, body, dataset_key)
            else:
                logits = model(ctx, body)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(labels.tolist())

    n_classes = len(label_names)
    class_idx = list(range(n_classes))

    acc    = accuracy_score(trues, preds)
    f1_mac = f1_score(trues, preds, average="macro", labels=class_idx, zero_division=0)
    f1_per = f1_score(trues, preds, average=None, labels=class_idx, zero_division=0)

    cm = confusion_matrix(trues, preds, labels=class_idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    acc_per = cm_norm.diagonal()
    mean_acc = acc_per.mean()

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test mean acc: {mean_acc:.4f}")
    print(f"Test F1 macro: {f1_mac:.4f}")
    for lbl, a, f1 in zip(label_names, acc_per, f1_per):
        print(f"  {lbl:<12} acc={a:.4f}  F1={f1:.4f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"{tag}")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_prefix.parent / f"{save_prefix.name}_confusion.png", dpi=120)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"{tag} — (Recall)")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_prefix.parent / f"{save_prefix.name}_confusion_norm.png", dpi=120)
    plt.show()

    return {
        "accuracy":    acc,
        "mean_acc":    mean_acc,
        "f1_macro":    f1_mac,
        "f1_per_class":  dict(zip(label_names, f1_per.tolist())),
        "acc_per_class": dict(zip(label_names, acc_per.tolist())),
    }
