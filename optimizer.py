# -----------------------------------------------------------
# Natural-Evolution-Strategies optimiser for a patch tensor
# -----------------------------------------------------------
import torch


class PatchOptimizer:
    """
    Black-box optimiser for an adversarial patch.
    Implements centred Natural-Evolution-Strategies (OpenAI ES-style).
    """
    def __init__(
        self,
        patch,                 # Patch() instance (gives .patch parameter)
        lr          = 0.2,     # base learning-rate for NES step
        sigma       = 0.3,     # std-dev of exploration noise
        num_samples = 30,      # population size per iteration
        lambda_norm = 0.0      # perceptual L_inf penalty weight
    ):
        # -------- NEW / CHANGED -----------
        self.patch       = patch
        self.lr          = lr
        self.sigma       = sigma
        self.num_samples = 100          # <─  was 30 or 50
        self.top_k       = 10           # <─  keep 10 best noises
        # ----------------------------------

        self.lambda_norm = lambda_norm
        self.noise_batch = []

    # -------------------------------------------------------
    def propose(self):
        """Return a list[Tensor] of mutated patch candidates."""
        self.noise_batch = [
            torch.randn_like(self.patch.patch) for _ in range(self.num_samples)
        ]
        # detach so the optimiser never keeps graph clutter
        return [(self.patch.get() + self.sigma * n).detach()
                for n in self.noise_batch]

    # -------------------------------------------------------
    def update(self, rewards, *, bg_crop=None):
        """
        `rewards`: list[float] - higher is better (e.g. drop in target confidence).
        """
        # optional perceptual penalty (physical attack)
        if self.lambda_norm > 0 and bg_crop is not None:
            delta = self.patch.get() - bg_crop
            l_inf = torch.max(torch.abs(delta)).item()
            rewards = [r - self.lambda_norm * l_inf for r in rewards]

        rewards = torch.tensor(rewards, dtype=torch.float32)
        # —— NES gradient: centre + normalise rewards ——
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        grad = torch.zeros_like(self.patch.patch)
        for r, n in zip(rewards, self.noise_batch):
            grad += r * n
        grad /= (self.num_samples * self.sigma)

        # SGD step on the patch parameter (in-place, with clamping)
        self.patch.update(self.lr * grad)
