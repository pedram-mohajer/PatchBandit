# placement_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Grid of valid top-left patch placements.
# A cell is valid if its centre lies inside *any* GT box (overlap-allowed).
# ------------------------------------------------------------------
# placement_utils.py
import torch

def get_valid_placement_grid(img_shape, gt_boxes, patch_sz, grid_sz, allow_background_fallback=True):
    """
    Return a list of (x,y) top-left coords where a patch of size `patch_sz`
    could be placed. Original behaviour: keep cells whose *centre* falls
    inside any GT box. This can return an empty list when the patch grows
    large. We now guarantee at least one candidate if
    `allow_background_fallback=True` by tiling the whole image.

    Args:
        img_shape: (C,H,W)
        gt_boxes:  Tensor[N,4] or np array [[xmin,ymin,xmax,ymax],...]
        patch_sz:  int (square patch)
        grid_sz:   int (#cells per axis)
        allow_background_fallback: if True, add full-image fallback

    Returns:
        list[(x,y)]
    """
    _, H, W = img_shape
    if torch.is_tensor(gt_boxes):
        gtb = gt_boxes.detach().cpu().numpy()
    else:
        gtb = gt_boxes
    if gtb is None:
        gtb = []

    cell_h, cell_w = H // grid_sz, W // grid_sz
    valid = []
    for r in range(grid_sz):
        for c in range(grid_sz):
            x = c * cell_w
            y = r * cell_h
            # keep only cells that fully fit
            if x + patch_sz > W or y + patch_sz > H:
                continue
            # centre of this candidate
            # Make sure at least 25 % of the patch area is inside *some* GT box
            patch_box = np.array([x, y, x + patch_sz, y + patch_sz], dtype=float)
            inside = False
            for bx in gtb:
                inter = _inter_area(patch_box, bx)
                if inter / (patch_sz * patch_sz) >= 0.25:
                    inside = True
                    break
            if inside:
                valid.append((x, y))


    # ---- fallback: no GT-overlapping cells, tile the full image ----
    if not valid and allow_background_fallback:
        stride_h = max(1, (H - patch_sz) // max(grid_sz - 1, 1))
        stride_w = max(1, (W - patch_sz) // max(grid_sz - 1, 1))
        for r in range(grid_sz):
            y = min(r * stride_h, H - patch_sz)
            for c in range(grid_sz):
                x = min(c * stride_w, W - patch_sz)
                valid.append((x, y))
        # at this point valid guaranteed non-empty

    return valid


# ------------------------------------------------------------------
# === NEW === Context features for bandit
# For each candidate cell, compute simple numeric features:
#   [norm_center_dist_to_nearest_gt, frac_overlap_gt, local_color_var, onehot_border]
# Return numpy array shape (D,)
# ------------------------------------------------------------------
def cell_context_features(cell, patch_sz, img_tensor, gt_boxes):
    x, y = cell
    _, H, W = img_tensor.shape
    cx, cy = x + patch_sz / 2, y + patch_sz / 2

    # nearest GT centre distance
    if len(gt_boxes) > 0:
        gcenters = np.stack(
            [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in gt_boxes], axis=0
        )
        dists = np.sqrt((gcenters[:, 0] - cx) ** 2 + (gcenters[:, 1] - cy) ** 2)
        d_min = float(dists.min())
    else:
        d_min = float(np.sqrt(H * H + W * W))
    d_norm = d_min / np.sqrt(H * H + W * W)

    # fractional overlap (approx via area intersection with nearest GT)
    frac_overlap = 0.0
    if len(gt_boxes) > 0:
        # patch box
        bx = np.array([x, y, x + patch_sz, y + patch_sz], dtype=float)
        overlaps = []
        for gt in gt_boxes:
            inter = _inter_area(bx, gt)
            gt_area = max(1.0, (gt[2] - gt[0]) * (gt[3] - gt[1]))
            overlaps.append(inter / gt_area)
        frac_overlap = float(max(overlaps)) if overlaps else 0.0

    # local color variance (stealth indicator)
    ph, pw = patch_sz, patch_sz
    color_var = 0.0
    if y + ph <= H and x + pw <= W:
        crop = img_tensor[:, y : y + ph, x : x + pw]
        color_var = float(crop.var().item())

    # border proximity one-hot
    margin = patch_sz / 2
    near_border = int(x < margin or y < margin or (x + patch_sz) > (W - margin) or (y + patch_sz) > (H - margin))

    return np.array([d_norm, frac_overlap, color_var, near_border], dtype=np.float32)


def _inter_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ------------------------------------------------------------------
def apply_patch_to_image(image, patch, location, mode="digital", alpha=1.0, show=False):
    """
    image: CHW tensor in *normalised* space (ImageNet mean/std)
    patch: CHW tensor already in [0,1] range (digital) or [min,max] (physical)
    """
    x, y = location
    patched_image = image.clone()
    ph, pw = patch.shape[1:]

    if y + ph > image.shape[1] or x + pw > image.shape[2]:
        return image  # invalid placement

    background_crop = patched_image[:, y : y + ph, x : x + pw]
    merged = merge_patch_with_background(patch, background_crop, mode, alpha)
    patched_image[:, y : y + ph, x : x + pw] = merged

    if show:
        # Denormalise (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        img_np = (patched_image * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.title(f"Patched Image at ({x},{y})")
        plt.axis("off")
        plt.show()

    return patched_image


def merge_patch_with_background(patch, background, mode="digital", alpha=0.5):
    if mode == "digital":
        # Soft blend (alpha mix)
        return alpha * patch + (1 - alpha) * background
    elif mode == "physical":
        # Additive residual, clamp to displayable
        delta = patch - background
        merged = background + delta
        return torch.clamp(merged, 0.0, 1.0)
    else:
        return patch


# ------------------------------------------------------------------
# IoU helper
# ------------------------------------------------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


# ------------------------------------------------------------------
def centre_on_gt(gt_box, patch_size, H, W):
    """
    One (x,y) whose centre is at GT box centre; clipped to image.
    """
    xmin, ymin, xmax, ymax = map(int, gt_box)
    cx = (xmin + xmax) // 2 - patch_size // 2
    cy = (ymin + ymax) // 2 - patch_size // 2
    cx = max(0, min(cx, W - patch_size))
    cy = max(0, min(cy, H - patch_size))
    return [(cx, cy)]
