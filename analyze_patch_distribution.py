import torch
from tqdm import tqdm
import numpy as np

def analyze_patch_distribution(loader, max_batches=None):
    fg_ratios = []
    artery_ratios = []
    vein_ratios = []

    has_fg_list = []
    has_art_list = []
    has_vein_list = []

    total_patches = 0

    for bidx, batch in enumerate(tqdm(loader, desc="Analyzing patches")):
        if max_batches is not None and bidx >= max_batches:
            break

        lab = batch["label"]  # (B,1,D,H,W) or (B,D,H,W)

        if lab.ndim == 4:
            lab = lab.unsqueeze(1)

        lab = lab.long()

        B = lab.shape[0]

        # flatten each patch
        lab_flat = lab.view(B, -1)

        fg = (lab_flat > 0).float()
        art = (lab_flat == 1).float()
        vein = (lab_flat == 2).float()

        # ratio
        fg_ratio = fg.mean(dim=1)
        art_ratio = art.mean(dim=1)
        vein_ratio = vein.mean(dim=1)

        # presence
        has_fg = (fg.sum(dim=1) > 0).float()
        has_art = (art.sum(dim=1) > 0).float()
        has_vein = (vein.sum(dim=1) > 0).float()

        fg_ratios.extend(fg_ratio.cpu().numpy())
        artery_ratios.extend(art_ratio.cpu().numpy())
        vein_ratios.extend(vein_ratio.cpu().numpy())

        has_fg_list.extend(has_fg.cpu().numpy())
        has_art_list.extend(has_art.cpu().numpy())
        has_vein_list.extend(has_vein.cpu().numpy())

        total_patches += B

    # ===== 統計 =====
    def summarize(arr, name):
        arr = np.array(arr)
        print(f"\n[{name}]")
        print(f"mean = {arr.mean():.6f}")
        print(f"std  = {arr.std():.6f}")
        print(f"min  = {arr.min():.6f}")
        print(f"max  = {arr.max():.6f}")

    print("\n========== PATCH DISTRIBUTION ==========")
    print(f"Total patches: {total_patches}")

    summarize(fg_ratios, "Foreground ratio")
    summarize(artery_ratios, "Artery ratio")
    summarize(vein_ratios, "Vein ratio")

    print("\n========== PRESENCE ==========")
    print(f"has_fg   = {np.mean(has_fg_list):.4f}")
    print(f"has_art  = {np.mean(has_art_list):.4f}")
    print(f"has_vein = {np.mean(has_vein_list):.4f}")

    # ===== histogram（很重要）=====
    print("\n========== HISTOGRAM (fg_ratio) ==========")
    hist, bins = np.histogram(fg_ratios, bins=10)
    for i in range(len(hist)):
        print(f"{bins[i]:.4f} ~ {bins[i+1]:.4f} : {hist[i]}")