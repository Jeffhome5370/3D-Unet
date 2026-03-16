import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped

from train_hipas_monai_unet3d_5fold import (
    DATA_ROOT,
    VAL_ROI_SIZE,
    VAL_OVERLAP,
    HiPaSNPZDataset,
    list_case_ids,
    build_5fold_split,
)

# ===================================================
#FOLD_IDX = 0
CKPT_ROOT = "./ckpt_hipas_unet3d_5fold"
CT_NII_DIR = "/home/e118/Datasets/HiPaS_original/ct_scan(.nii.gz)"
SAVE_NIFTI = False
OUT_DIR = "./test_predictions_npz"
CACHE_RATE_TEST = 0.2
NUM_WORKERS = 1
# ====================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold index (0~4)"
    )

    return parser.parse_args()

def build_test_transform():
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])


def get_best_ckpt_path(fold_idx: int) -> str:
    ckpt_path = os.path.join(CKPT_ROOT, f"fold{fold_idx}", "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    return ckpt_path


def get_test_ids(fold_idx: int):
    case_ids = list_case_ids(DATA_ROOT)
    _, _, test_ids = build_5fold_split(case_ids, fold_idx)
    return test_ids


def ct_npz_path_from_case_id(cid: str) -> str:
    p = os.path.join(DATA_ROOT, f"{cid}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到對應 CT npz：{p}")
    return p


def save_pred_npz(pred_dhw: np.ndarray, cid: str) -> str:
    src_npz = np.load(ct_npz_path_from_case_id(cid))

    if "image" not in src_npz:
        raise KeyError(f"[{cid}] npz 裡沒有 'image' key")

    image = src_npz["image"]

    # 處理可能的 channel 維度
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if pred_dhw.shape != image.shape:
        raise RuntimeError(
            f"[{cid}] pred shape {pred_dhw.shape} != image shape {image.shape}\n"
            f"代表 prediction 與原始 npz 影像尺寸不一致。"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{cid}_pred.npz")

    np.savez_compressed(
        out_path,
        pred=pred_dhw.astype(np.uint8)
    )
    return out_path


def build_model(device: torch.device) -> UNet:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


def main():
    args = get_args()
    FOLD_IDX = args.fold
    print(f"Running fold {FOLD_IDX}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = get_best_ckpt_path(FOLD_IDX)
    test_ids = get_test_ids(FOLD_IDX)
    print(f"Fold {FOLD_IDX} | Test cases: {len(test_ids)}")

    test_ds = CacheDataset(
        data=HiPaSNPZDataset(DATA_ROOT, test_ids, transform=build_test_transform()),
        cache_rate=CACHE_RATE_TEST,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loaded:", ckpt_path)
    print("Best epoch:", ckpt.get("epoch", "N/A"))
    print("Best val dice:", ckpt.get("best_dice", "N/A"))

    dice_overall = DiceMetric(include_background=False, reduction="mean")
    dice_per_class = DiceMetric(include_background=False, reduction="none")

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.inference_mode(), torch.amp.autocast(autocast_device):
        for batch in tqdm(test_loader, desc=f"Fold {FOLD_IDX} [test]"):
            img = batch["image"].to(device)
            lab = batch["label"].to(device).long()
            if lab.ndim == 4:
                lab = lab.unsqueeze(1)

            pred = sliding_window_inference(
                inputs=img,
                roi_size=VAL_ROI_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=VAL_OVERLAP,
            )

            pred_cls = pred.argmax(dim=1)
            lab_idx = lab.squeeze(1)

            pred_cls_cpu = pred_cls.detach().cpu()
            lab_idx_cpu = lab_idx.detach().cpu()

            pred_onehot = F.one_hot(pred_cls_cpu, num_classes=3).permute(0, 4, 1, 2, 3).float()
            lab_onehot = F.one_hot(lab_idx_cpu, num_classes=3).permute(0, 4, 1, 2, 3).float()

            dice_overall(pred_onehot, lab_onehot)
            dice_per_class(pred_onehot, lab_onehot)

            if SAVE_NIFTI:
                cid = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else batch["case_id"]
                save_pred_npz(pred_cls_cpu[0].numpy().astype(np.uint8), cid)

            del pred, pred_cls, lab_idx
            if device.type == "cuda":
                torch.cuda.empty_cache()

    mean_dice = float(dice_overall.aggregate().item())
    pc = dice_per_class.aggregate()
    pc_mean = torch.nanmean(pc, dim=0)
    artery_dice = float(pc_mean[0].item())
    vein_dice = float(pc_mean[1].item())

    print("\n========= TEST RESULT =========")
    print(f"Fold        : {FOLD_IDX}")
    print(f"Mean Dice   : {mean_dice:.4f}")
    print(f"Artery Dice : {artery_dice:.4f}")
    print(f"Vein Dice   : {vein_dice:.4f}")


if __name__ == "__main__":
    main()