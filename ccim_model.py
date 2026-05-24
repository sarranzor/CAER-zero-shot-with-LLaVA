#### Utils, CCIM module and complete DoubleResnet model for training and evaluation

### Imports
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm.auto import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix,
)

from dual_resnet_shared import (
    device,
    FEAT_CTX_DIM, FEAT_BODY_DIM, JOINT_DIM,
    EMOTIC_LABELS, NCAERS_LABELS,
    LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, GRAD_CLIP,
    ContextEncoder, BodyEncoder,
)

### Config
NUM_CONF = 1024
CCIM_STRATEGY = "dp_cause"


### CCIM https://github.com/ydk122024/CCIM

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class dot_product_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super().__init__()
        self.con_size = con_size
        self.query = nn.Linear(fuse_size, 256, bias=False)
        self.key = nn.Linear(con_size, 256, bias=False)

    def forward(self, confounder_set, fuse_rep, probabilities):
        query = self.query(fuse_rep)
        key = self.key(confounder_set)
        mid = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.con_size)
        attention = F.softmax(mid, dim=-1).unsqueeze(2)
        return (attention * confounder_set * probabilities).sum(1)


class additive_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super().__init__()
        self.query = nn.Linear(fuse_size, 256, bias=False)
        self.key = nn.Linear(con_size, 256, bias=False)
        self.w_t = nn.Linear(256, 1, bias=False)

    def forward(self, confounder_set, fuse_rep, probabilities):
        query  = self.query(fuse_rep).unsqueeze(1)
        key = self.key(confounder_set)
        fuse = torch.tanh(query + key)
        attention = F.softmax(self.w_t(fuse), dim=1)
        return (attention * confounder_set * probabilities).sum(1)


class _CCIMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128 * 4)
        self.fc2 = nn.Linear(128 * 4, 128)
        self.drop = nn.Dropout(p=0.5)
        self.norm = nn.BatchNorm1d(128)

    def forward(self, out):
        residual = out
        out = self.norm(out)
        out = gelu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return residual + out * 0.3


class CCIM(nn.Module):
    def __init__(self, num_joint_feature, num_gz, strategy):
        super().__init__()
        self.num_joint_feature = num_joint_feature
        self.num_gz = num_gz
        if strategy == "dp_cause":
            self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature)
        elif strategy == "ad_cause":
            self.causal_intervention = additive_intervention(num_gz, num_joint_feature)
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        self.w_h  = Parameter(torch.empty(num_joint_feature, 128))
        self.w_g = Parameter(torch.empty(num_gz, 128))
        self.classifier = _CCIMClassifier()
        self.emotic_fc = nn.Linear(128, len(EMOTIC_LABELS))
        self.caers_fc = nn.Linear(128, len(NCAERS_LABELS))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

    def forward(self, joint_feature, confounder_dictionary, prior, dataset):
        g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior)
        proj_h = torch.matmul(joint_feature, self.w_h)
        proj_g_z = torch.matmul(g_z, self.w_g)
        do_x = proj_h + proj_g_z
        out = self.classifier(do_x)
        if dataset == "EMOTIC":
            return self.emotic_fc(out)
        elif dataset in ("CAER_S", "SynthCAER"):
            return self.caers_fc(out)
        raise ValueError(f"Unknown dataset: {dataset}")

### Cofounder dict

@torch.no_grad()
def build_confounder_dictionary(model, loader, n_clusters=NUM_CONF, cache_path=None):
    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        print(f"Dict loaded in cache: {cache_path}")
        return data["centers"], data["prior"]

    model.eval()
    all_feats = []
    for batch in tqdm(loader, desc="Extracting features:"):
        feats = model.extract_context_features(batch[0].to(device)).cpu().numpy()
        all_feats.append(feats)

    all_feats = np.vstack(all_feats)
    print(f"Features extracted: {all_feats.shape}")

    kmeans = MiniBatchKMeans(
        n_clusters = n_clusters, random_state=42, n_init=5,
        batch_size = min(4096, len(all_feats)), verbose=0,
    )
    kmeans.fit(all_feats)

    counts = np.bincount(kmeans.labels_, minlength=n_clusters).astype(np.float32)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    prior = torch.tensor(counts / counts.sum(), dtype=torch.float32).unsqueeze(1)

    if cache_path:
        torch.save({"centers": centers, "prior": prior}, cache_path)
        print(f"Dict saved in: {cache_path}")

    return centers, prior

### Complete Model

class FinalModel(nn.Module):

    def __init__(
        self,
        ctx_dim = FEAT_CTX_DIM,
        body_dim = FEAT_BODY_DIM,
        joint_dim = JOINT_DIM,
        num_conf = NUM_CONF,
        strategy = CCIM_STRATEGY,
        freeze_backbone = False,
    ):
        super().__init__()
        self.ctx_encoder = ContextEncoder(ctx_dim, freeze_backbone)
        self.body_encoder = BodyEncoder(body_dim, freeze_backbone)
        self.fusion = nn.Sequential(
            nn.Linear(ctx_dim + body_dim, joint_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(joint_dim),
        )
        self.ccim = CCIM(joint_dim, ctx_dim, strategy)
        self.register_buffer("confounder_dict",  torch.zeros(num_conf, ctx_dim))
        self.register_buffer("confounder_prior", torch.ones(num_conf, 1) / num_conf)

    def encode(self, context_img, body_img):
        ctx_feat = self.ctx_encoder(context_img)
        body_feat = self.body_encoder(body_img)
        return self.fusion(torch.cat([ctx_feat, body_feat], dim=1))

    def forward(self, context_img, body_img, dataset):
        joint = self.encode(context_img, body_img)
        return self.ccim(joint, self.confounder_dict, self.confounder_prior, dataset)

    @torch.no_grad()
    def extract_context_features(self, context_img):
        return self.ctx_encoder(context_img)

    def set_confounders(self, centers, prior):
        self.confounder_dict.copy_(centers)
        self.confounder_prior.copy_(prior)

### Training

def train_single_label(
    model,
    train_loader,
    val_loader,
    class_weights,
    dataset_key,
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
            logits = model(ctx, body, dataset_key)
            loss   = criterion(logits, labels)
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
                logits = model(ctx, body, dataset_key)
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

def evaluate_single_label(
    model,
    loader,
    dataset_key,
    label_names,
    tag,
    save_prefix,
):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ctx, body, labels in tqdm(loader, desc=f"Evaluating {tag}:"):
            logits = model(ctx.to(device), body.to(device), dataset_key)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(labels.tolist())

    n_classes = len(label_names)
    class_idx = list(range(n_classes))
    acc = accuracy_score(trues, preds)
    f1_mac = f1_score(trues, preds, average="macro", labels=class_idx, zero_division=0)
    f1_per = f1_score(trues, preds, average=None, labels=class_idx, zero_division=0)

    cm = confusion_matrix(trues, preds, labels=class_idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    acc_per = np.diag(cm_norm)
    mean_acc = acc_per.mean()

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test mean acc: {mean_acc:.4f}")
    print(f"Test F1 macro: {f1_mac:.4f}")
    for lbl, a, f1 in zip(label_names, acc_per, f1_per):
        print(f"{lbl:<12} acc={a:.4f}  F1={f1:.4f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Double ResNet + CCIM — {tag}")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_prefix.parent / f"{save_prefix.name}_confusion.png", dpi=120)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Double ResNet + CCIM — {tag} (Recall)")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_prefix.parent / f"{save_prefix.name}_confusion_norm.png", dpi=120)
    plt.show()

    return {
        "accuracy": acc,
        "mean_acc": mean_acc,
        "f1_macro": f1_mac,
        "f1_per_class": dict(zip(label_names, f1_per.tolist())),
        "acc_per_class": dict(zip(label_names, acc_per.tolist())),
    }


def plot_training_curves(history, tag: str, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax2 = ax.twinx()
    ax2.plot(history["val_f1"], color="green",  linestyle="--", label="Val F1")
    ax2.plot(history["val_acc"], color="orange", linestyle=":", label="Val mean-acc")
    ax2.set_ylabel("Métrica", color="green")
    ax.set_title(f"Entrenamiento Double ResNet + CCIM — {tag}")
    ax.set_xlabel("Época"); ax.set_ylabel("Loss")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()