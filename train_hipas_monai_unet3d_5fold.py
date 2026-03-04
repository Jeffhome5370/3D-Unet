import os
import glob
import random
import csv
import json
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from monai.data import CacheDataset, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    DeleteItemsd,
    SpatialPadd,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# ====================== 你要改的設定 ======================
DATA_ROOT = "/home/e118/Datasets/HiPaS_original"  # 你的資料根目錄（包含 ct_scan(.npz)/ artery(.npz)/ vein(.npz)）
SEED = 42

# HU normalize
HU_CLIP_MIN = -1000
HU_CLIP_MAX = 2000

# patch 訓練設定
PATCH_SIZE = (64, 192, 192)      # (D, H, W)
PATCH_SAMPLES_PER_CASE = 4       # 每個 case 抽幾個 patch（RandCropByLabelClassesd 的 num_samples）

# validation sliding window
VAL_ROI_SIZE = (96, 192, 192)
VAL_OVERLAP = 0.25

# 训练超参
EPOCHS = 500
VAL_EVERY = 10
LR = 2e-4
WEIGHT_DECAY = 1e-5

# loader
BATCH_SIZE = 4
NUM_WORKERS = 4

# cache（RAM 夠可調高）
CACHE_RATE_TRAIN = 0.2
CACHE_RATE_VAL = 0.2

# 5-fold
N_FOLDS = 5
TEST_FIXED_RANGE = (200, 250)  # 以排序後 index 計：case_ids[200:250] -> 201~250 (若檔名是001..250)
RUN_ALL_FOLDS = True           # True: 跑 fold0~4；False: 只跑 SINGLE_FOLD
SINGLE_FOLD = 0                # 0~4

# 輸出
CKPT_ROOT = "./ckpt_hipas_unet3d_5fold"
LOG_ROOT = "./logs_hipas_unet3d_5fold"
# =========================================================

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_logger(log_dir: str, fold_idx: int) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"hipas_fold{fold_idx}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重複打印

    # 清掉舊 handler（例如 notebook 重跑）
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh_path = os.path.join(log_dir, f"fold{fold_idx}_{ts}.log")
    fh = logging.FileHandler(fh_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. logfile={fh_path}")
    return logger


def load_npz_data(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)["data"]


def list_case_ids(data_root: str):
    ct_files = sorted(glob.glob(os.path.join(data_root, "ct_scan", "*.npz")))
    if len(ct_files) == 0:
        raise FileNotFoundError(f"找不到 ct_scan/*.npz：{os.path.join(data_root,'ct_scan')}")
    case_ids = [os.path.splitext(os.path.basename(p))[0] for p in ct_files]
    return case_ids


def build_5fold_split(case_ids, fold_idx: int, seed: int = 42):
    """
    250 筆：固定最後 50 筆(201~250)為 test；前 200 筆做 5-fold（每 fold val=40, train=160）
    注意：這裡「201~250」是以 case_ids 排序後的位置推斷，請確保你的檔名排序為 001..250。
    """
    case_ids = sorted(case_ids)
    n_total = len(case_ids)
    if n_total < 250:
        raise ValueError(f"case_ids len={n_total} < 250，請確認資料是否完整。")
    if n_total > 250:
        # 你若真的只有 250 筆，這邊可改成 raise；這裡保守只取前 250
        case_ids = case_ids[:250]
        n_total = 250

    test_start, test_end = TEST_FIXED_RANGE  # 200, 250
    trainval_ids = case_ids[:test_start]     # 0..199 => 200 筆
    test_ids = case_ids[test_start:test_end] # 200..249 => 50 筆

    assert len(trainval_ids) == 200, f"trainval_ids={len(trainval_ids)}"
    assert len(test_ids) == 50, f"test_ids={len(test_ids)}"

    ids = trainval_ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)

    fold_size = len(ids) // N_FOLDS  # 40
    if fold_size * N_FOLDS != len(ids):
        raise ValueError("200 無法整除 5？這不該發生，請檢查 N_FOLDS。")

    if not (0 <= fold_idx < N_FOLDS):
        raise ValueError(f"fold_idx must be in [0,{N_FOLDS-1}]")

    val_start = fold_idx * fold_size
    val_end = val_start + fold_size

    val_ids = ids[val_start:val_end]
    train_ids = ids[:val_start] + ids[val_end:]

    return train_ids, val_ids, test_ids


class HiPaSNPZDataset:
    """
    讀取 HiPaS npz：
      ct_scan(.npz)/{id}.npz
      artery(.npz)/artery/{id}.npz
      vein(.npz)/vein/{id}.npz

    輸出 dict:
      image: float32, (D,H,W) -> transform 加 channel -> (1,D,H,W)
      label: int64,  (D,H,W), 0=bg 1=artery 2=vein
    """
    def __init__(self, data_root: str, case_ids, transform=None):
        self.data_root = data_root
        self.case_ids = case_ids
        self.transform = transform

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        ct_path = os.path.join(self.data_root, "ct_scan", f"{cid}.npz")
        a_path = os.path.join(self.data_root, "annotation", "artery", f"{cid}.npz")
        v_path = os.path.join(self.data_root, "annotation", "vein", f"{cid}.npz")

        ct = load_npz_data(ct_path)      # (H,W,D) e.g. (512,512,258)
        artery = load_npz_data(a_path)   # (H,W,D) 0/1
        vein = load_npz_data(v_path)     # (H,W,D) 0/1

        ct = np.ascontiguousarray(np.asarray(ct)).astype(np.float32, copy=False)
        artery = (np.asarray(artery) > 0).astype(np.uint8)
        vein = (np.asarray(vein) > 0).astype(np.uint8)

        label = np.zeros_like(artery, dtype=np.int64)
        label[artery == 1] = 1
        label[vein == 1] = 2  # overlap: vein 覆蓋 artery（可自行改）

        # (H,W,D) -> (D,H,W)
        ct = np.transpose(ct, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # HU clip + normalize -> [0,1]
        ct = np.clip(ct, HU_CLIP_MIN, HU_CLIP_MAX).astype(np.float32)
        ct = (ct - HU_CLIP_MIN) / float(HU_CLIP_MAX - HU_CLIP_MIN)

        sample = {"image": ct, "label": label, "case_id": cid}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_transforms():
    train_tf = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"], track_meta=False),

        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),

        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            ratios=[0.0, 1.0, 1.0],     # 0=bg,1=artery,2=vein
            num_classes=3,
            num_samples=PATCH_SAMPLES_PER_CASE,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, method="end"),
        DeleteItemsd(keys=["case_id"]),
    ])

    val_tf = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])
    return train_tf, val_tf


def poly_lr(epoch):
    power = 0.9
    return (1 - epoch / EPOCHS) ** power


def warmup_poly(epoch):
    warmup_epochs = int(EPOCHS * 0.05)
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        return poly_lr(epoch)


def loss_fn(pred, target, device):
    # target: (B,1,D,H,W) label map
    if target.ndim == 6 and target.shape[2] == 1:
        target = target.squeeze(2)

    if target.ndim == 5 and target.shape[1] > 1:
        target = target.argmax(dim=1, keepdim=True)

    class_weights = torch.tensor([0.1, 1.0, 1.0], device=device)
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    loss_dice = dice_loss(pred, target)
    loss_ce = ce_loss(pred, target.squeeze(1).long())
    return loss_dice + loss_ce


def write_epoch_csv_row(csv_path: str, row: dict, header_order: list):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_one_fold(fold_idx: int):
    # 固定 determinism（也可加 fold offset，避免每 fold augmentation 序列完全一樣）
    set_determinism(SEED + fold_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dirs
    fold_dir = os.path.join(CKPT_ROOT, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    logger = setup_logger(LOG_ROOT, fold_idx)

    # list + split
    case_ids = list_case_ids(DATA_ROOT)
    train_ids, val_ids, test_ids = build_5fold_split(case_ids, fold_idx, seed=SEED)

    logger.info(f"device={device}")
    logger.info(f"Total={len(case_ids)} | train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    logger.info(f"Test fixed ids (first/last): {test_ids[:3]} ... {test_ids[-3:]}")

    train_tf, val_tf = build_transforms()
    train_data = HiPaSNPZDataset(DATA_ROOT, train_ids, transform=train_tf)
    val_data = HiPaSNPZDataset(DATA_ROOT, val_ids, transform=val_tf)

    train_ds = CacheDataset(train_data, cache_rate=CACHE_RATE_TRAIN, num_workers=NUM_WORKERS)
    val_ds = CacheDataset(val_data, cache_rate=CACHE_RATE_VAL, num_workers=max(1, NUM_WORKERS // 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(opt, lr_lambda=warmup_poly)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    dice_overall = DiceMetric(include_background=False, reduction="mean")   # artery+vein 平均
    dice_per_class = DiceMetric(include_background=False, reduction="none")# (artery, vein)

    best_dice = -1.0
    best_epoch = -1
    csv_path = os.path.join(fold_dir, "metrics.csv")
    header = ["fold", "epoch", "lr", "train_loss", "val_mean_dice", "val_artery_dice", "val_vein_dice", "best_dice_so_far"]

    logger.info(f"Start training fold={fold_idx}. ckpt_dir={fold_dir}")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fold {fold_idx} | Epoch {epoch} [train]"):
            
            img = batch["image"].to(device)                 # (B,1,D,H,W)
            lab = batch["label"].to(device).long()          # (B,D,H,W) or (B,1,D,H,W) after transform
            if lab.ndim == 4:
                lab = lab.unsqueeze(1)                      # (B,1,D,H,W)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(img)                         # (B,3,D,H,W)
                loss = loss_fn(logits, lab, device=device)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item())
            
        running_loss /= max(1, len(train_loader))
        current_lr = float(opt.param_groups[0]["lr"])
        logger.info(f"[Fold {fold_idx}] Epoch {epoch} train_loss={running_loss:.6f} lr={current_lr:.6g}")

        # scheduler after epoch
        scheduler.step()

        val_mean = None
        artery = None
        vein = None

        if epoch % VAL_EVERY == 0:
            model.eval()
            dice_overall.reset()
            dice_per_class.reset()
            
            with torch.inference_mode(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                for batch in tqdm(val_loader, desc=f"Fold {fold_idx} | Epoch {epoch} [val]"):
                    img = batch["image"].to(device)              # (1,1,D,H,W)
                    lab = batch["label"].to(device).long()       # (1,D,H,W) or (1,1,D,H,W)
                    if lab.ndim == 4:
                        lab = lab.unsqueeze(1)                   # (1,1,D,H,W)

                    pred = sliding_window_inference(
                        inputs=img,
                        roi_size=VAL_ROI_SIZE,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=VAL_OVERLAP,
                    )  # (1,3,D,H,W)

                    pred_cls = pred.argmax(dim=1)  # (B,D,H,W)
                    pred_d = F.one_hot(pred_cls, num_classes=3).permute(0, 4, 1, 2, 3).float()
                    lab_idx = lab.squeeze(1).long()
                    lab_d = F.one_hot(lab_idx, num_classes=3).permute(0, 4, 1, 2, 3).float()

                    dice_overall(pred_d, lab_d)
                    dice_per_class(pred_d, lab_d)
        
            
            val_mean = float(dice_overall.aggregate().item())
            pc = dice_per_class.aggregate()              # (B,2) 累積
            pc_mean = torch.nanmean(pc, dim=0)           # (2,)
            artery = float(pc_mean[0].item())
            vein = float(pc_mean[1].item())

            logger.info(f"[Fold {fold_idx}] Epoch {epoch} VAL mean_dice={val_mean:.6f} artery={artery:.6f} vein={vein:.6f}")

            # save best on val mean
            if val_mean > best_dice:
                best_dice = val_mean
                best_epoch = epoch
                ckpt_path = os.path.join(fold_dir, "best.pth")
                torch.save(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_dice": best_dice,
                        "config": {
                            "LR": LR,
                            "WEIGHT_DECAY": WEIGHT_DECAY,
                            "BATCH_SIZE": BATCH_SIZE,
                            "PATCH_SIZE": PATCH_SIZE,
                            "VAL_ROI_SIZE": VAL_ROI_SIZE,
                            "VAL_OVERLAP": VAL_OVERLAP,
                            "EPOCHS": EPOCHS,
                            "VAL_EVERY": VAL_EVERY,
                            "SEED": SEED,
                        },
                        "split": {
                            "train_ids": train_ids,
                            "val_ids": val_ids,
                            "test_ids": test_ids,
                        },
                    },
                    ckpt_path,
                )
                logger.info(f"✅ Saved best checkpoint: {ckpt_path} (best_dice={best_dice:.6f} @ epoch={best_epoch})")

            torch.cuda.empty_cache()

        # per-epoch csv（val 沒跑就留空）
        write_epoch_csv_row(
            csv_path,
            {
                "fold": fold_idx,
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": running_loss,
                "val_mean_dice": "" if val_mean is None else val_mean,
                "val_artery_dice": "" if artery is None else artery,
                "val_vein_dice": "" if vein is None else vein,
                "best_dice_so_far": best_dice,
            },
            header_order=header,
        )
    
    # fold summary
    summary = {
        "fold": fold_idx,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
    }
    with open(os.path.join(fold_dir, "fold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Fold {fold_idx} finished. best_dice={best_dice:.6f} best_epoch={best_epoch}")
    return summary


def main():
    os.makedirs(CKPT_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)

    summaries = []
    folds = list(range(N_FOLDS)) if RUN_ALL_FOLDS else [SINGLE_FOLD]

    for fold_idx in folds:
        summaries.append(run_one_fold(fold_idx))

    # overall stats
    bests = [s["best_dice"] for s in summaries]
    mean_best = float(np.mean(bests))
    std_best = float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0

    overall = {
        "n_folds": len(summaries),
        "fold_summaries": summaries,
        "best_dice_mean": mean_best,
        "best_dice_std": std_best,
    }

    with open(os.path.join(CKPT_ROOT, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # 也存一份 csv summary
    csv_sum = os.path.join(CKPT_ROOT, "cv_summary.csv")
    with open(csv_sum, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "best_dice", "best_epoch", "train_count", "val_count", "test_count"])
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print("==== 5-fold CV finished ====")
    print("Best dice per fold:", bests)
    print(f"Mean best dice: {mean_best:.6f}")
    print(f"Std  best dice: {std_best:.6f}")
    print(f"Saved: {os.path.join(CKPT_ROOT,'cv_summary.json')} and {csv_sum}")


if __name__ == "__main__":
    main()
