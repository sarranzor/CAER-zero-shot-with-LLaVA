#### Utils, encoders and datasets shared by Double ResNet notebooks

### Imports
import os, random, urllib.request, warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

from sklearn.model_selection import train_test_split

### Config
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Paths
EMOTIC_ROOT = Path("datasets/emotic")
EMOTIC_PARQUET = Path("emotic_basic.parquet")

NCAERS_ROOT = Path("datasets/NCAERS")
NCAERS_INDEX = NCAERS_ROOT / "ncaers_index.parquet"

SYNTH_ROOT = Path("datasets/SynthCAER")
SYNTH_INDEX = SYNTH_ROOT / "synthcontext_index.parquet"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

## Hyperparams
FEAT_CTX_DIM = 256
FEAT_BODY_DIM = 256
JOINT_DIM = 256

BATCH_SIZE = 32
NUM_WORKERS = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
GRAD_CLIP = 1.0

## Labels
EMOTIC_LABELS = ["Anger", "Aversion", "Fear", "Happiness", "Sadness", "Surprise"]
EMOTIC_TO_IDX = {lbl: i for i, lbl in enumerate(EMOTIC_LABELS)}

NCAERS_LABELS = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NCAERS_TO_IDX = {lbl: i for i, lbl in enumerate(NCAERS_LABELS)}

SYNTH_LABEL_MAP = {
    "Angry": "Anger", "Disgust": "Disgust", "Fear": "Fear",
    "Happy": "Happy", "Neutral": "Neutral", "Sad": "Sad", "Surprise": "Surprise",
}
SYNTH_TO_IDX = {s: NCAERS_TO_IDX[n] for s, n in SYNTH_LABEL_MAP.items()}


### Encoders
# Load Places365
def load_places365_resnet50():
    model_path = MODELS_DIR / "resnet50_places365.pth.tar"
    if not model_path.exists():
        url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
        urllib.request.urlretrieve(url, model_path)
        print("Places365 downloaded")
    base  = models.resnet50(num_classes=365)
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    base.load_state_dict(state)
    return base

# Delete last layer
def backbone_without_fc(model):
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten())

## ContextEncoder (Places365)
class ContextEncoder(nn.Module):
    def __init__(self, out_dim = FEAT_CTX_DIM, freeze_backbone = False):
        super().__init__()
        self.backbone = backbone_without_fc(load_places365_resnet50())
        print("Places 365 loaded.")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim), nn.ReLU(inplace=True), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))

## BodyEncoder (ImageNet)
class BodyEncoder(nn.Module):
    def __init__(self, out_dim = FEAT_BODY_DIM, freeze_backbone = False):
        super().__init__()
        self.backbone = backbone_without_fc(models.resnet50(pretrained=True))
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim), nn.ReLU(inplace=True), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


### Face detection (Haar cascade) -> For BodyEncoder with NCAER-S & SynthcAER
def _get_cascade_path():
    filename = "haarcascade_frontalface_default.xml"
    try:
        p = os.path.join(cv2.data.haarcascades, filename)
        if os.path.isfile(p):
            return p
    except AttributeError:
        pass
    p = os.path.join(os.path.dirname(cv2.__file__), "data", filename)
    if os.path.isfile(p):
        return p
    local = Path(filename)
    if not local.exists():
        url = ("https://raw.githubusercontent.com/opencv/opencv/master/"
               "data/haarcascades/" + filename)
        urllib.request.urlretrieve(url, local)
    return str(local)


_face_cascade = cv2.CascadeClassifier(_get_cascade_path())


def detect_face_crop(img, padding = 0.15):
    W, H = img.size
    gray = np.array(img.convert("L"))
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    # Fallback (Full image)
    if not len(faces):
        return img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * padding), int(h * padding)
    return img.crop((max(0, x - pad_x), max(0, y - pad_y),
                     min(W, x + w + pad_x), min(H, y + h + pad_y)))


### Class weights

def compute_class_weights(labels_series, n_classes):
    counts = labels_series.value_counts().reindex(range(n_classes), fill_value=1).values.astype(float)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

### Datasets
## EMOTIC
class EMOTICDataset(Dataset):
    def __init__(self, df, img_root, transform_ctx, transform_body):
        self.df, self.img_root = df.reset_index(drop=True), img_root
        self.transform_ctx  = transform_ctx
        self.transform_body = transform_body

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_root / row["folder"] / row["filename"]).convert("RGB")
        W, H = img.size
        x1 = max(0, min(int(row["body_x1"]), W - 1))
        y1 = max(0, min(int(row["body_y1"]), H - 1))
        x2 = max(x1 + 1, min(int(row["body_x2"]), W))
        y2 = max(y1 + 1, min(int(row["body_y2"]), H))
        body_crop = img.crop((x1, y1, x2, y2))
        label = EMOTIC_TO_IDX[row["basic_emotion"]]
        return self.transform_ctx(img), self.transform_body(body_crop), label


def make_emotic_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf_train  = T.Compose([
        T.Resize((256, 256)), T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.CenterCrop(224),
        T.ToTensor(), normalize,
    ])
    tf_val = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

    df = pd.read_parquet(EMOTIC_PARQUET)
    df = df[df["basic_emotion"].isin(EMOTIC_LABELS)].copy()

    df_train, df_tmp = train_test_split(df, test_size=0.30, stratify=df["basic_emotion"], random_state=42)
    df_val,  df_test = train_test_split(df_tmp, test_size=0.50, stratify=df_tmp["basic_emotion"], random_state=42)

    print("EMOTIC distribution (70/15/15):")
    print(pd.concat([df_train.assign(split="train"), df_val.assign(split="val"),
                     df_test.assign(split="test")])
          .groupby(["split", "basic_emotion"]).size().unstack(fill_value=0))

    ds_train = EMOTICDataset(df_train, EMOTIC_ROOT, tf_train, tf_train)
    ds_val = EMOTICDataset(df_val, EMOTIC_ROOT, tf_val, tf_val)
    ds_test = EMOTICDataset(df_test, EMOTIC_ROOT, tf_val, tf_val)

    loader_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    df_train = df_train.copy()
    df_train["label_idx"] = df_train["basic_emotion"].map(EMOTIC_TO_IDX)
    weights = compute_class_weights(df_train["label_idx"], len(EMOTIC_LABELS))

    return loader_train, loader_val, loader_test, weights

## NCAER-S
class NCaersDataset(Dataset):
    def __init__(self, df, root, transform_ctx, transform_body):
        self.df, self.root = df.reset_index(drop=True), root
        self.transform_ctx = transform_ctx
        self.transform_body = transform_body

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["split"] / row["label"] / row["filename"]).convert("RGB")
        label = NCAERS_TO_IDX[row["label"]]
        face_crop = detect_face_crop(img)
        return self.transform_ctx(img), self.transform_body(face_crop), label


def make_ncaers_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf_train  = T.Compose([
        T.Resize((256, 256)), T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.CenterCrop(224),
        T.ToTensor(), normalize,
    ])
    tf_val = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

    df = pd.read_parquet(NCAERS_INDEX)
    df["split"] = df["split"].replace({"validation": "val"})
    df["label_idx"] = df["label"].map(NCAERS_TO_IDX)

    print("NCAERS distribution:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    ds_train = NCaersDataset(splits["train"], NCAERS_ROOT, tf_train, tf_train)
    ds_val = NCaersDataset(splits["val"], NCAERS_ROOT, tf_val, tf_val)
    ds_test = NCaersDataset(splits["test"], NCAERS_ROOT, tf_val, tf_val)

    loader_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    weights = compute_class_weights(splits["train"]["label_idx"], len(NCAERS_LABELS))

    return loader_train, loader_val, loader_test, weights

## SynthCAER
# SynthCAER index
def build_synth_index(root = SYNTH_ROOT, cache= SYNTH_INDEX, seed = 42):
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"Index loaded: {len(df):,} images")
        print(df.groupby(["split", "emotion"]).size().unstack(fill_value=0))
        return df

    records = []
    valid   = set(SYNTH_TO_IDX.keys())
    for seed_dir in sorted(root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for ctx_dir in sorted(seed_dir.iterdir()):
            if not ctx_dir.is_dir():
                continue
            for img_file in sorted(ctx_dir.glob("*.png")):
                if img_file.stem not in valid:
                    continue
                records.append({
                    "seed": seed_dir.name,
                    "context": ctx_dir.name,
                    "emotion": img_file.stem,
                    "filename": img_file.name,
                    "rel_path": img_file.relative_to(root).as_posix(),
                    "label_idx": SYNTH_TO_IDX[img_file.stem],
                })

    df = pd.DataFrame(records)
    rng = np.random.default_rng(seed)
    parts = []
    for (s, _), grp in df.groupby(["seed", "emotion"]):
        if s != "seed_52":
            grp = grp.copy(); grp["split"] = "train"
        else:
            idx = rng.permutation(len(grp))
            sp = np.array(["val"] * len(grp), dtype=object)
            sp[idx[len(grp) // 2:]] = "test"
            grp = grp.copy(); grp["split"] = sp
        parts.append(grp)

    df = pd.concat(parts).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"Index built: {len(df):,} images")
    print(df.groupby(["split", "emotion"]).size().unstack(fill_value=0))
    return df

class SynthContextDataset(Dataset):
    def __init__(self, df, root, transform_ctx, transform_body):
        self.df, self.root = df.reset_index(drop=True), root
        self.transform_ctx = transform_ctx
        self.transform_body = transform_body

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["rel_path"]).convert("RGB")
        crop = detect_face_crop(img)
        return self.transform_ctx(img), self.transform_body(crop), int(row["label_idx"])


def make_synth_loaders(df=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    if df is None:
        df = build_synth_index()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf_train  = T.Compose([
        T.Resize((256, 256)), T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.CenterCrop(224),
        T.ToTensor(), normalize,
    ])
    tf_val = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    ds_train = SynthContextDataset(splits["train"], SYNTH_ROOT, tf_train, tf_train)
    ds_val = SynthContextDataset(splits["val"], SYNTH_ROOT, tf_val, tf_val)
    ds_test = SynthContextDataset(splits["test"], SYNTH_ROOT, tf_val, tf_val)

    loader_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    weights = compute_class_weights(splits["train"]["label_idx"], len(NCAERS_LABELS))

    return loader_train, loader_val, loader_test, weights