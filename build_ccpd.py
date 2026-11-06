"""Step 1 (CCPD part): build clean plate crops + ground truth.

Usage:
    python build_ccpd.py --root /path/to/CCPD2019 --out data
Outputs:
    data/ccpd_plates.npz   (base_imgs, base_text, blur_imgs, blur_text)
    data/ccpd_plates.csv   (id, source, true_text, filename)
    data/ccpd_preview.png  (grid for visual check)
"""
import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np

# CCPD character tables (verify against the CCPD README before trusting).
# Index 0 (province) is Chinese and is dropped from the ground truth.
ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]

W, H, MARGIN = 240, 76, 8
GREY = 128  # 8 px grey margin (same as synthetic plates; match step 1.1)


def parse_name(name):
    """Return (corners 4x2 float32, true_text) or None if malformed."""
    f = Path(name).stem.split("-")
    if len(f) < 7:
        return None
    try:
        corners = np.array([[int(v) for v in p.split("&")]
                            for p in f[3].split("_")], dtype=np.float32)
        idx = [int(i) for i in f[4].split("_")]
        if corners.shape != (4, 2) or len(idx) != 7:
            return None
        chars = [ALPHABETS[idx[1]]] + [ADS[i] for i in idx[2:]]
    except (ValueError, IndexError):
        return None
    text = "".join(c for c in chars if c != "O")
    return corners, text


def load_plate(path):
    parsed = parse_name(path.name)
    if parsed is None:
        return None
    corners, text = parsed
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # CCPD vertex order starts bottom-right, then goes around (check visually!)
    dst = np.array([[W - 1, H - 1], [0, H - 1], [0, 0], [W - 1, 0]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    plate = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)
    plate = cv2.copyMakeBorder(plate, MARGIN, MARGIN, MARGIN, MARGIN,
                               cv2.BORDER_CONSTANT, value=GREY)
    return plate.astype(np.float32) / 255.0, text


def sample(folder, n, seed):
    files = sorted(folder.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(files)
    imgs, texts, names = [], [], []
    for p in files:
        out = load_plate(p)
        if out is None:
            continue
        imgs.append(out[0]); texts.append(out[1]); names.append(p.name)
        if len(imgs) == n:
            break
    return np.stack(imgs), np.array(texts), names


def preview(imgs, texts, path, n=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 4, figsize=(12, 8))
    for ax, im, t in zip(axes.ravel(), imgs[:n], texts[:n]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_base", type=int, default=400)
    ap.add_argument("--n_blur", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    b_img, b_txt, b_names = sample(root / "ccpd_base", a.n_base, a.seed)
    r_img, r_txt, r_names = sample(root / "ccpd_blur", a.n_blur, a.seed)

    np.savez_compressed(out / "ccpd_plates.npz",
                        base_imgs=b_img, base_text=b_txt,
                        blur_imgs=r_img, blur_text=r_txt)
    with open(out / "ccpd_plates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source", "true_text", "filename"])
        for i, (t, n) in enumerate(zip(b_txt, b_names)):
            w.writerow([f"ccpd_base_{i:04d}", "ccpd_base", t, n])
        for i, (t, n) in enumerate(zip(r_txt, r_names)):
            w.writerow([f"ccpd_blur_{i:04d}", "ccpd_blur", t, n])

    preview(b_img, b_txt, out / "ccpd_preview.png")
    print(f"base: {len(b_img)}  blur: {len(r_img)}  shape: {b_img.shape[1:]}")
