# patch_bandit.py
"""
Adaptive black-box adversarial patch attack with:
  • Contextual placement bandit (Linear-UCB) or ε-greedy grid
  • NES-style patch parameter optimisation
  • Budget-aware patch size growth
  • Joint reward: detector suppression + (optional) stealth + printability
  • EOT simulation for physical robustness
  • FAST MODE: early-exit & watchdogs
"""

import os, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm

from placement_utils import get_valid_placement_grid, cell_context_features
from patch           import Patch
from bandit          import EpsGridBandit, ContextualUCBPlacementBandit
from dataloader      import VOC_CLASSES


# ================================================================
# FAST / WATCHDOG CONTROLS
# ================================================================
STOP_ON_TOP_SWAP       = True     # stop when detector top-1 ≠ target
TOP_SWAP_STREAK        = 1        # require N consecutive swaps
PRINT_EVERY_STEPS      = 1
MAX_TIME_PER_IMAGE_SEC = 60
MAX_STEPS_THIS_SIZE    = 20
# ================================================================


# ----------------------------------------------------------------
# NES Optimiser (single, fixed implementation)
# ----------------------------------------------------------------
class PatchOptimizer:
    """Centred-NES gradient-free optimiser for a learnable patch tensor."""
    def __init__(
        self,
        patch,
        lr=0.2,
        sigma=0.3,
        num_samples=30,
        lambda_norm=0.0,
        top_k=None,
    ):
        self.patch        = patch
        self.lr           = lr
        self.sigma        = sigma
        self.num_samples  = num_samples           # <- honours config
        self.top_k        = top_k or num_samples  # keep all by default
        self.lambda_norm  = lambda_norm
        self.noise_batch  = []

    def propose(self):
        base = self.patch.get().detach()
        self.noise_batch = [torch.randn_like(base) for _ in range(self.num_samples)]
        return [(base + self.sigma * n).clamp(0, 1).detach() for n in self.noise_batch]

    def step(self, rewards):
        if not rewards:
            return
        r = torch.tensor(rewards, dtype=torch.float32)
        r = (r - r.mean()) / (r.std() + 1e-8)

        # keep only top-k noises to reduce variance
        idx_keep = torch.topk(r, self.top_k).indices.tolist() if self.top_k < len(r) else range(len(r))

        grad = torch.zeros_like(self.patch.patch)
        for i in idx_keep:
            grad += float(r[i]) * self.noise_batch[i].to(grad.device)
        grad /= (len(idx_keep) * self.sigma)

        self.patch.update(self.lr * grad)


# ----------------------------------------------------------------
# Attack configuration dataclass
# ----------------------------------------------------------------
@dataclass
class AttackConfig:
    patch_sizes: Tuple[int, ...] = (40, 60, 80, 120, 160)
    grow_patience: int = 50
    min_improv: float = 0.0
    max_queries: int = 250
    bandit_eps: float = 0.2

    w_det: float = 1.0
    w_stealth: float = 0.0
    w_print: float = 0.0

    grid_size: int = 8
    use_contextual: bool = True
    ucb_alpha: float = 1.0

    nes_lr: float = 0.7
    nes_sigma: float = 0.4
    nes_pop: int = 20
    nes_topk: int = 10

    eot_samples: int = 0
    eot_scale_jitter: float = 0.25
    eot_rotate_deg: float  = 15.0
    eot_color_jitter: float = 0.05

    success_conf_drop: float = 0.25
    success_abs_conf:  float = 0.10

    save_every: int = 0
    save_dir:   str = "results"


# ----------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------
def _to_uint8(image_t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    img = (image_t * std + mean).permute(1, 2, 0).clamp(0, 1)
    return (img.detach().cpu().numpy() * 255).astype(np.uint8)


def _top1(boxes):
    if boxes is None or len(boxes) == 0:
        return -1, 0.0
    idx = int(torch.argmax(boxes.conf).item())
    return int(boxes.cls[idx].item()), float(boxes.conf[idx].item())


def _conf_for_cls(boxes, cls_id: int):
    if boxes is None or len(boxes) == 0:
        return 0.0
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    return float(conf_arr[cls_arr == cls_id].max()) if cls_id in cls_arr else 0.0


def _print_boxes_by_class(boxes):
    if boxes is None or len(boxes) == 0:
        print("[CLEAN DETECTOR] No detections.")
        return {}
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    per_cls = {}
    for c, s in zip(cls_ids, confs):
        per_cls[c] = max(per_cls.get(c, 0.0), float(s))
    for c, s in sorted(per_cls.items(), key=lambda x: x[1], reverse=True):
        name = VOC_CLASSES[c] if 0 <= c < len(VOC_CLASSES) else str(c)
        print(f"  → {name} ({s:.3f})")
    return per_cls


def stealth_penalty(patch_px, bg_px):
    return torch.mean(torch.abs(patch_px - bg_px)).item()


def printability_penalty(patch_px):
    p = patch_px.clamp(0, 1)
    return float(torch.relu(0.1 - p).mean() + torch.relu(p - 0.9).mean() + 0.25 * p.std())


def eot_augment(image_px, device, scale_j=0.25, rot_deg=15.0, color_j=0.05):
    C, H, W = image_px.shape
    out = image_px.unsqueeze(0)

    # scale jitter
    if scale_j > 0:
        s = float(np.random.uniform(1.0 - scale_j, 1.0 + scale_j))
        nh, nw = max(1, int(H * s)), max(1, int(W * s))
        out = torch.nn.functional.interpolate(out, size=(nh, nw), mode="bilinear", align_corners=False)
        if nh >= H and nw >= W:
            y0 = (nh - H) // 2; x0 = (nw - W) // 2
            out = out[:, :, y0:y0+H, x0:x0+W]
        else:
            pad_h = max(0, H - nh); pad_w = max(0, W - nw)
            out = torch.nn.functional.pad(out, (0, pad_w, 0, pad_h))
            out = out[:, :, :H, :W]

    # crude rotation via roll
    if rot_deg > 0:
        max_shift = int((rot_deg / 45.0) * min(H, W) * 0.1)
        if max_shift > 0:
            dy = np.random.randint(-max_shift, max_shift + 1)
            dx = np.random.randint(-max_shift, max_shift + 1)
            out = torch.roll(out, shifts=(dy, dx), dims=(2, 3))

    # colour jitter
    if color_j > 0:
        out = (out + torch.randn_like(out) * color_j).clamp(0, 1)

    return out.squeeze(0).to(device)


# ----------------------------------------------------------------
# PatchAttackRunner
# ----------------------------------------------------------------
class PatchAttackRunner:
    def __init__(
        self,
        image_tensor: torch.Tensor,
        gt_boxes,
        gt_labels,
        *,
        min_size=20,
        max_size=100,
        grow_every=50,
        grid_size=8,
        attack_type="digital",
        lambda_norm=0.0,
        use_bandit=True,
        attack_cfg: Optional[AttackConfig] = None,
        save_prefix: str = "results/picture",
    ):
        # store clean, normalised image
        self.image = image_tensor.clone()   # CHW, ImageNet norm
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.mode = attack_type
        self.save_prefix = save_prefix
        self.device = self.image.device
        self.C, self.H, self.W = self.image.shape

        # normalisation constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3,1,1)

        # build config from legacy args if needed
        if attack_cfg is None:
            attack_cfg = AttackConfig(
                patch_sizes=(min_size, max_size),
                grow_patience=grow_every,
                w_det=1.0,
                w_stealth=0.0,
                w_print=0.0,
                grid_size=grid_size,
                use_contextual=use_bandit,
                max_queries=250,
            )
        self.cfg = attack_cfg

        # first patch
        self._make_patch(self.cfg.patch_sizes[0], lambda_norm=self.cfg.w_print)

        # bandit
        if self.cfg.use_contextual:
            self.bandit = ContextualUCBPlacementBandit(feat_dim=4, alpha=self.cfg.ucb_alpha, min_pulls=1)
        else:
            #self.bandit = EpsGridBandit(epsilon=0.2)
            self.bandit = EpsGridBandit(epsilon=self.cfg.bandit_eps)

        # runtime state
        self.history = []  # (step,size,plain_conf,eot_conf,reward,cell,top_cls,top_conf)
        self.best_reward = -1e9
        self.stalled_steps = 0
        self.cur_size_index = 0

    # -------------------- norm / pixel --------------------
    def _to_px(self, img_norm: torch.Tensor) -> torch.Tensor:
        return (img_norm * self.std + self.mean).clamp(0,1)

    def _to_norm(self, img_px: torch.Tensor) -> torch.Tensor:
        return (img_px - self.mean) / self.std

    # -------------------- safe paste ----------------------
    def _apply_patch_normsafe(self, img_norm: torch.Tensor, patch_01: torch.Tensor, loc):
        x, y = loc
        ph, pw = patch_01.shape[1:]
        img_px = self._to_px(img_norm).clone()
        img_px[:, y:y+ph, x:x+pw] = patch_01.to(img_px.device).clamp(0,1)
        return self._to_norm(img_px)

    # -------------------- EOT avg -------------------------
    def _avg_eot_conf(self, model, img_norm: torch.Tensor, cls_id: int) -> float:
        if self.cfg.eot_samples <= 0:
            conf, _ = self._model_conf(model, img_norm, cls_id)
            return conf
        conf_acc = 0.0
        img_px = self._to_px(img_norm)
        for _ in range(self.cfg.eot_samples):
            aug_px = eot_augment(
                img_px.clone(), self.device,
                scale_j=self.cfg.eot_scale_jitter,
                rot_deg=self.cfg.eot_rotate_deg,
                color_j=self.cfg.eot_color_jitter,
            )
            aug_norm = self._to_norm(aug_px)
            c, _ = self._model_conf(model, aug_norm, cls_id)
            conf_acc += c
        return conf_acc / max(1, self.cfg.eot_samples)

    # -------------------- build patch ---------------------
    def _make_patch(self, size, lambda_norm=0.0):
        self.patch_size = size
        self.patch = Patch(size=size, mode=self.mode, device=self.device, init="rand")
        self.optimizer = PatchOptimizer(
            self.patch,
            lr=self.cfg.nes_lr,
            sigma=self.cfg.nes_sigma,
            num_samples=self.cfg.nes_pop,
            lambda_norm=lambda_norm,
            top_k=self.cfg.nes_topk,
        )
        self.valid_cells = get_valid_placement_grid(self.image.shape, self.gt_boxes, size, self.cfg.grid_size)
        if not self.valid_cells:
            self.valid_cells = self._fallback_cells()
            print(f"[WARN] no grid cells at size {size}; using fallback {self.valid_cells}")

    # -------------------- grow patch ----------------------
    def _resize_patch(self, new_size):
        src = self.patch.get().detach()
        self._make_patch(new_size, lambda_norm=self.cfg.w_print)
        with torch.no_grad():
            up = F.interpolate(src.unsqueeze(0), size=(new_size, new_size), mode="bilinear", align_corners=False).squeeze(0)
            self.patch.patch.copy_(up)

    # -------------------- fallback cells ------------------
    def _fallback_cells(self):
        # prefer largest GT box centre
        if (self.gt_boxes is not None) and (len(self.gt_boxes) > 0):
            gtb = self.gt_boxes.detach().cpu().numpy() if torch.is_tensor(self.gt_boxes) else self.gt_boxes
            areas = (gtb[:,2]-gtb[:,0]) * (gtb[:,3]-gtb[:,1])
            i = int(np.argmax(areas))
            xmin, ymin, xmax, ymax = gtb[i]
            cx = int((xmin + xmax) / 2.0) - self.patch_size // 2
            cy = int((ymin + ymax) / 2.0) - self.patch_size // 2
        else:
            cx = (self.W - self.patch_size) // 2
            cy = (self.H - self.patch_size) // 2
        cx = max(0, min(cx, self.W - self.patch_size))
        cy = max(0, min(cy, self.H - self.patch_size))
        return [(cx, cy)]

    # -------------------- pick cell -----------------------
    def _pick_cell(self):
        if not self.valid_cells:
            self.valid_cells = self._fallback_cells()
        if isinstance(self.bandit, ContextualUCBPlacementBandit):
            return self.bandit.select_arm(
                self.valid_cells,
                lambda c: cell_context_features(c, self.patch_size, self.image, self.gt_boxes),
            )
        else:
            return self.bandit.select_arm(self.valid_cells)

    # -------------------- model conf ----------------------
    def _to_uint8(self, img_t):
        return _to_uint8(img_t, self.mean, self.std)

    def _model_conf(self, model, img_t, cls_id):
        boxes = model.predict(self._to_uint8(img_t), conf=0.25, verbose=False)[0].boxes
        return _conf_for_cls(boxes, cls_id), boxes

    # -------------------- reward total --------------------
    def _reward_total(self, conf_plain, conf_eot, patch_px, bg_px):
        det_gain_plain = self.clean_conf_plain - conf_plain
        det_gain_eot   = self.clean_conf_eot   - conf_eot
        det_gain = 0.5 * (det_gain_plain + det_gain_eot)
        st_p = stealth_penalty(patch_px, bg_px) if self.cfg.w_stealth > 0 else 0.0
        pr_p = printability_penalty(patch_px)  if self.cfg.w_print   > 0 else 0.0
        return self.cfg.w_det * det_gain - self.cfg.w_stealth * st_p - self.cfg.w_print * pr_p

    # -------------------- growth patience -----------------
    def _maybe_grow(self):
        if self.cur_size_index + 1 >= len(self.cfg.patch_sizes):
            return False
        self.stalled_steps += 1
        if self.stalled_steps < self.cfg.grow_patience:
            return False
        new_size = self.cfg.patch_sizes[self.cur_size_index + 1]
        print(f"[GROW] {self.patch_size} → {new_size}")
        self._resize_patch(new_size)
        self.cur_size_index += 1
        self.stalled_steps = 0
        return True

    # -------------------- save ----------------------------
    def _save(self, vis_img_norm, size, succeeded=False, step=None):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        tag = "succeeded" if succeeded else "failed"
        base = f"{self.save_prefix}_{tag}_patch_{size}"
        if step is not None:
            base += f"_step_{step}"
        idx = 0
        while os.path.exists(f"{base}_{idx}.png"):
            idx += 1
        torchvision.utils.save_image(self.patch.get().detach().cpu(), f"{base}_{idx}_patch.png")
        torchvision.utils.save_image(self._to_px(vis_img_norm).cpu(), f"{base}_{idx}.png")
        print(f"[✓] Saved → {base}_{idx}.png  &  {base}_{idx}_patch.png")

    # ---------- fast single-call EOT helper ----------   
    def _avg_eot_conf_px(self, model, img_px, cls_id):
        """Like _avg_eot_conf but expects pixel-space CHW tensor, to avoid extra RGB→norm→RGB shuttling."""
        if self.cfg.eot_samples <= 0:
            img_norm = self._to_norm(img_px)
            conf, _ = self._model_conf(model, img_norm, cls_id)
            return conf
        acc = 0.0
        for _ in range(self.cfg.eot_samples):
            aug = eot_augment(img_px.clone(), self.device,
                            self.cfg.eot_scale_jitter,
                            self.cfg.eot_rotate_deg,
                            self.cfg.eot_color_jitter)
            acc += self._model_conf(model, self._to_norm(aug), cls_id)[0]
        return acc / self.cfg.eot_samples

    # -------------------- run -----------------------------
    def run(self, model, *, max_iters=250, max_attempts=1):
        # clean detector (low thresh diag)
        clean_pred = model.predict(self._to_uint8(self.image), conf=0.001, verbose=False)[0].boxes
        _print_boxes_by_class(clean_pred)
        tgt_cls, clean_conf_plain = _top1(clean_pred)
        if tgt_cls == -1:
            tgt_cls = int(self.gt_labels[0])
            clean_conf_plain = 1.0
        print(f"[INFO] Target = {VOC_CLASSES[tgt_cls]} (clean_conf={clean_conf_plain:.2f})")

        self.clean_conf_plain = clean_conf_plain
        self.clean_conf_eot = self._avg_eot_conf(model, self.image, tgt_cls) if self.cfg.eot_samples > 0 else clean_conf_plain
        self.image_px = self._to_px(self.image)

        step_global      = 0
        success          = False
        vis_img_norm     = None
        top_swap_streak  = 0
        steps_this_size  = 0
        current_size_tag = self.patch_size
        t_start          = time.time()

        pbar = tqdm(total=max_iters, desc=f"opt:{VOC_CLASSES[tgt_cls]}@{self.patch_size}", leave=False, dynamic_ncols=True,disable=True)

        while step_global < max_iters:
            # wallclock watchdog
            if MAX_TIME_PER_IMAGE_SEC > 0 and (time.time() - t_start) > MAX_TIME_PER_IMAGE_SEC:
                print(f"[FAST-BYPASS] Time > {MAX_TIME_PER_IMAGE_SEC}s. Aborting image.")
                break

            # per-size watchdog
            # ---------- per-size watchdog ----------
            if steps_this_size >= MAX_STEPS_THIS_SIZE:
                bigger_exists = self.cur_size_index + 1 < len(self.cfg.patch_sizes)
                if bigger_exists:
                    print(f"[FAST-GROW] {MAX_STEPS_THIS_SIZE} steps ↦ grow patch.")
                    self._resize_patch(self.cfg.patch_sizes[self.cur_size_index + 1])
                    self.cur_size_index += 1
                    steps_this_size = 0
                    current_size_tag = self.patch_size
                    pbar.set_description_str(f"opt:{VOC_CLASSES[tgt_cls]}@{self.patch_size}")
                    pbar.refresh()
                    continue          # retry with new size
                else:
                    print("[FAST-BYPASS] Largest size reached; aborting image.")
                    break

            # placement
            cell = self._pick_cell()
            x, y = cell
            if (x + self.patch_size > self.W) or (y + self.patch_size > self.H):
                # shouldn't happen, but skip
                continue

            # background crop
            bg_px = self.image_px[:, y:y+self.patch_size, x:x+self.patch_size]

            # candidate population
            pop = self.optimizer.propose()
            rewards, confs_plain = [], []

            for cand in pop:
                img_c_norm = self._apply_patch_normsafe(self.image, cand, (x, y))
                conf_plain, _ = self._model_conf(model, img_c_norm, tgt_cls)
                rewards.append(self._reward_total(conf_plain, conf_plain, cand, bg_px))  # eot conf placeholder
                confs_plain.append(conf_plain)

            # NES update uses plain-image reward (fast)
            self.optimizer.step(rewards)

            # ----- Evaluate best candidate with EOT only once -----
            best_i  = int(np.argmax(rewards))
            best_p  = pop[best_i]
            vis_img_norm = self._apply_patch_normsafe(self.image, best_p, (x, y))  # <- keep same name
            best_conf_plain = confs_plain[best_i]
            best_conf_eot   = self._avg_eot_conf_px(model, self._to_px(vis_img_norm), tgt_cls)
            best_r          = self._reward_total(best_conf_plain, best_conf_eot, best_p, bg_px)

            # top-1 on patched plain
            boxes_plain = model.predict(self._to_uint8(vis_img_norm), conf=0.001, verbose=False)[0].boxes

            top_cls_plain, top_conf_plain = _top1(boxes_plain)
            top_name = "None" if top_cls_plain == -1 else (
                VOC_CLASSES[top_cls_plain] if 0 <= top_cls_plain < len(VOC_CLASSES) else str(top_cls_plain)
            )

            # log step
            if PRINT_EVERY_STEPS > 0 and (step_global % PRINT_EVERY_STEPS == 0):
                print(
                    f"[A|S{step_global:4}] top={top_name}({top_conf_plain:.3f}) "
                    f"target={best_conf_plain:.3f} eot={best_conf_eot:.3f} r={best_r:+.3f}"
                )
            pbar.update(1)

            # top-swap fast exit
            if STOP_ON_TOP_SWAP:
                if top_cls_plain != tgt_cls:
                    top_swap_streak += 1
                else:
                    top_swap_streak = 0
                if top_swap_streak >= TOP_SWAP_STREAK:
                    print(
                        f"[SUCCESS:top-swap] Step {step_global}: "
                        f"top={top_name}({top_conf_plain:.3f}) "
                        f"target={VOC_CLASSES[tgt_cls]}({best_conf_plain:.3f})"
                    )
                    self.patch.patch.data.copy_(best_p.to(self.patch.patch.device))
                    self.history.append((step_global, self.patch_size, best_conf_plain, best_conf_eot, best_r, cell, top_cls_plain, top_conf_plain))
                    self._save(vis_img_norm, self.patch_size, succeeded=True, step=step_global)
                    pbar.close()
                    return True, step_global, {"final_size": self.patch_size, "history": self.history}

            # bandit feedback
            if isinstance(self.bandit, ContextualUCBPlacementBandit):
                feat_vec = cell_context_features(cell, self.patch_size, self.image, self.gt_boxes)
                self.bandit.update(cell, best_r, feat_vec)
            else:
                self.bandit.update(cell, best_r)

            # growth (patience)
            if best_r > self.best_reward + self.cfg.min_improv:
                self.best_reward = best_r
                self.stalled_steps = 0
            else:
                grew = self._maybe_grow()
                if grew:
                    if not self.valid_cells:
                        self.valid_cells = self._fallback_cells()
                    steps_this_size  = 0
                    current_size_tag = self.patch_size
                    pbar.set_description_str(f"opt:{VOC_CLASSES[tgt_cls]}@{self.patch_size}")
                    continue

            # per-size watchdog step count
            if self.patch_size == current_size_tag:
                steps_this_size += 1
            else:
                current_size_tag = self.patch_size
                steps_this_size = 0

            # history
            self.history.append((step_global, self.patch_size, best_conf_plain, best_conf_eot, best_r, cell, top_cls_plain, top_conf_plain))
            if self.cfg.save_every > 0 and step_global % self.cfg.save_every == 0:
                self._save(vis_img_norm, self.patch_size, succeeded=False, step=step_global)

            # strict success (plain suppression)
            drop_plain = self.clean_conf_plain - best_conf_plain
            if (drop_plain >= self.cfg.success_conf_drop) and (best_conf_plain <= self.cfg.success_abs_conf):
                print(
                    f"[SUCCESS] Step {step_global}: plain_conf {best_conf_plain:.3f} "
                    f"(drop {drop_plain:.3f}) size {self.patch_size}"
                )
                self.patch.patch.data.copy_(best_p.to(self.patch.patch.device))
                self._save(vis_img_norm, self.patch_size, succeeded=True, step=step_global)
                success = True
                break

            step_global += 1

        # end loop
        pbar.close()
        if not success and vis_img_norm is not None:
            self._save(vis_img_norm, self.patch_size, succeeded=False, step=step_global)

        return success, step_global, {"final_size": self.patch_size, "history": self.history}