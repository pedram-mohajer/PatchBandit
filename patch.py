# patch.py
import torch
import torch.nn as nn


class Patch(nn.Module):
    """
    Learnable CHW patch tensor.
    Supports digital (0..1 clamp) and physical (print-friendly gamut clamp).
    """
    def __init__(self, size=40, mode="digital", device=None, init="rand"):
        super().__init__()
        self.size = size
        self.mode = mode
        self.device = device if device is not None else torch.device("cpu")
        if init == "gray":
            init_tensor = torch.full((3, size, size), 0.5, device=self.device)
        else:
            init_tensor = torch.rand(3, size, size, device=self.device)
        self.patch = nn.Parameter(init_tensor)

        # printer-safe bounds (tunable)
        self.phys_min = 0.1
        self.phys_max = 0.9

    def get(self):
        return self.apply_constraints(self.patch)

    def apply_constraints(self, patch_tensor):
        if self.mode == "digital":
            return torch.clamp(patch_tensor, 0.0, 1.0)
        elif self.mode == "physical":
            return torch.clamp(patch_tensor, self.phys_min, self.phys_max)
        else:
            return patch_tensor

    def update(self, delta):
        with torch.no_grad():
            self.patch += delta.to(self.patch.device)
            self.patch.copy_(self.apply_constraints(self.patch))

    def resize_from(self, new_size, src_tensor=None):
        """
        Resize patch param in-place to new_size (bilinear upsample).
        Optionally copy values from src_tensor (CHW) else self.get().
        """
        if src_tensor is None:
            src_tensor = self.get().detach()
        src = src_tensor.unsqueeze(0)
        up = torch.nn.functional.interpolate(
            src, size=(new_size, new_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        with torch.no_grad():
            self.size = new_size
            self.patch = nn.Parameter(up.to(self.device))
