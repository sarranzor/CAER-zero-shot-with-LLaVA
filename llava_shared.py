#### Utils, encoders and datasets shared by LLaVA notebooks

### Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 15

### LLaVA Load
def load_llava_model(model_id = MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NF4 quantization
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    # Model if GPU not avaiable
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    if torch.cuda.is_available():
        print(f"VRAM used: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
    else:
        print("Modelo in CPU")

    return model, processor, device

### LLaVA Output parsing for evaluation
def parse_emotion(response, emotions, aliases = None):
    reply = response.split("ASSISTANT:")[-1].strip()
    for emotion in emotions:
        if emotion.lower() in reply.lower():
            return emotion
    if aliases:
        reply_lower = reply.lower()
        for alias, canonical in aliases.items():
            if alias in reply_lower:
                return canonical
    return None

### LLaVA evaluation
def evaluate_llava_results(df_res,emotions,tag,save_stem,):
    df_ev = df_res[~df_res["unparseable"]].copy()
    n_unp = df_res["unparseable"].sum()

    
    # Metrics
    acc = accuracy_score(df_ev["true_emotion"], df_ev["predicted_emotion"])
    mf1 = f1_score(df_ev["true_emotion"], df_ev["predicted_emotion"],
                      labels=emotions, average="macro", zero_division=0)
    f1_per = f1_score(df_ev["true_emotion"], df_ev["predicted_emotion"],
                      labels=emotions, average=None, zero_division=0)

    acc_per = recall_score(df_ev["true_emotion"], df_ev["predicted_emotion"],
                           labels=emotions, average=None, zero_division=0)
    acc_bal = acc_per.mean()

    print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"Avg Acc. per class: {acc_bal:.3f}")
    for emo, a in zip(emotions, acc_per):
        print(f"Acc {emo:<12}: {a:.3f}")
    print(f"Macro F1: {mf1:.3f}")
    for emo, f in zip(emotions, f1_per):
        print(f"F1 {emo:<12}: {f:.3f}")
    print()
    print(classification_report(
        df_ev["true_emotion"], df_ev["predicted_emotion"],
        labels=emotions, zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(df_ev["true_emotion"], df_ev["predicted_emotion"], labels=emotions)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=emotions, yticklabels=emotions,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"LLaVA — {tag}  |  acc={acc:.3f}  F1={mf1:.3f}")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_stem}_confusion_counts.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=emotions, yticklabels=emotions,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"LLaVA — {tag} (Recall)")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_stem}_confusion_norm.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "accuracy": acc,
        "acc_balanced": acc_bal,
        "acc_per_class": dict(zip(emotions, acc_per.tolist())),
        "f1_macro": mf1,
        "f1_per_class": dict(zip(emotions, f1_per.tolist())),
        "unparseable_pct": n_unp / len(df_res) * 100,
    }

# Evaluation for robustness analysis
def run_kfold(df_kfold, emotions, n_folds, random_seed= 42,):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_records = []

    # Metrics per fold
    for fold, (_, val_idx) in enumerate(skf.split(df_kfold, df_kfold["true_emotion"]), start=1):
        df_fold = df_kfold.iloc[val_idx]

        acc     = accuracy_score(df_fold["true_emotion"], df_fold["predicted_emotion"])
        mf1     = f1_score(df_fold["true_emotion"], df_fold["predicted_emotion"],
                           labels=emotions, average="macro",    zero_division=0)
        clf1    = f1_score(df_fold["true_emotion"], df_fold["predicted_emotion"],
                           labels=emotions, average=None,       zero_division=0)
        acc_per = recall_score(df_fold["true_emotion"], df_fold["predicted_emotion"],
                               labels=emotions, average=None,   zero_division=0)
        acc_bal = acc_per.mean()

        row = {"Fold": fold, "N": len(df_fold), "Accuracy": acc,
               "Balanced Accuracy": acc_bal, "Macro F1": mf1}
        for emo, f1_val in zip(emotions, clf1):
            row[f"F1_{emo}"] = f1_val
        for emo, a_val in zip(emotions, acc_per):
            row[f"Acc_{emo}"] = a_val
        fold_records.append(row)

    return pd.DataFrame(fold_records)


def plot_kfold_results(df_folds, emotions, title, save_prefix,):
    n_folds = len(df_folds)
    _COLORS = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#FFC000",
               "#7030A0", "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B"]
    colors  = _COLORS[:len(emotions)]

    # Line plot
    fig, ax = plt.subplots(figsize=(8, 5))
    folds_x = df_folds["Fold"].astype(str)
    for metric, color in [("Accuracy", "#4472C4"), ("Macro F1", "#C00000")]:
        ax.plot(folds_x, df_folds[metric], marker="o", label=metric, color=color)
    ax.axhline(df_folds["Macro F1"].mean(), color="#C00000", linestyle="--", alpha=0.5,
               label=f"Media Macro F1 ({df_folds["Macro F1"].mean():.3f})")
    ax.set_xlabel("Fold"); ax.set_ylabel("Valor")
    ax.set_title("Métricas globales por fold")
    ax.legend(fontsize=9); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_kfold_lines.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Bar plot
    f1_cols = [f"F1_{e}" for e in emotions]
    means   = df_folds[f1_cols].mean().values
    stds    = df_folds[f1_cols].std().values
    fig, ax = plt.subplots(figsize=(8, 5))
    bars    = ax.bar(range(len(emotions)), means, yerr=stds, capsize=5,
                     color=colors, alpha=0.85, error_kw={"elinewidth": 1.5})
    ax.set_xticks(list(range(len(emotions))))
    ax.set_xticklabels(emotions, rotation=20, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title(f"F1 por clase — media ± std ({n_folds} folds)")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_kfold_bars.png", dpi=150, bbox_inches="tight")
    plt.show()
