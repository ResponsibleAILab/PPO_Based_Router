from __future__ import annotations

import os
import random
import threading
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# ────────────────────────────────────────────────────────────
# Simple epsilon-greedy baseline (kept for ablations)
# ────────────────────────────────────────────────────────────
class EpsilonGreedyPolicy:
    """Selects backend with lowest latency proxy with epsilon exploration."""
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.buffer = []

    def select(self, obs: List[List[float]]) -> int:
        # obs is per-backend rows: [mean, p95, queue, ...]
        if random.random() < self.epsilon:
            return random.randrange(len(obs))
        # score = mean + 0.5 * p95 + 0.1 * queue  (lower is better)
        scores = [o[0] + 0.5 * o[1] + 0.1 * o[2] for o in obs]
        return int(min(range(len(scores)), key=lambda i: scores[i]))

    # old signature kept for backwards compatibility (not used by app.py)
    def observe(self, o, a, r):
        self.buffer.append((o, a, r))
        if len(self.buffer) > 10_000:
            self.buffer = self.buffer[-5_000:]

    # new unified interface for bandit-style updates (no-op for epsilon)
    def update(self, backend_idx: int, reward: float) -> None:
        # epsilon-greedy baseline in this project is latency-based, so we
        # ignore the reward here to keep behavior identical.
        return

# ────────────────────────────────────────────────────────────
# Shared bandit base for Softmax / UCB / Thompson Sampling
# ────────────────────────────────────────────────────────────
class _BanditStatsBase:
    """
    Keeps per-backend running stats of reward.
    Uses Welford's algorithm for mean / variance.
    """
    def __init__(self):
        self.counts: List[int] = []
        self.means: List[float] = []
        self.m2: List[float] = []  # sum of squared diffs

    def _ensure_arms(self, n_arms: int) -> None:
        while len(self.counts) < n_arms:
            self.counts.append(0)
            self.means.append(0.0)
            self.m2.append(0.0)

    def update(self, backend_idx: int, reward: float) -> None:
        self._ensure_arms(backend_idx + 1)
        c = self.counts[backend_idx] + 1
        mean = self.means[backend_idx]
        delta = reward - mean
        mean_new = mean + delta / c
        delta2 = reward - mean_new
        m2_new = self.m2[backend_idx] + delta * delta2

        self.counts[backend_idx] = c
        self.means[backend_idx] = mean_new
        self.m2[backend_idx] = m2_new

    def _variance(self, backend_idx: int) -> float:
        c = self.counts[backend_idx]
        if c < 2:
            return 1.0
        return self.m2[backend_idx] / (c - 1)

class SoftmaxPolicy(_BanditStatsBase):
    """
    Softmax (Boltzmann) exploration over mean reward.
    Higher reward backends get higher probability.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def select(self, obs: List[List[float]]) -> int:
        n = len(obs)
        self._ensure_arms(n)

        # Ensure each arm is tried at least once
        for i in range(n):
            if self.counts[i] == 0:
                return i

        # Compute softmax over mean rewards
        temps = [m / max(self.temperature, 1e-8) for m in self.means[:n]]
        max_temp = max(temps)
        exps = [math.exp(x - max_temp) for x in temps]
        total = sum(exps)
        probs = [e / total for e in exps]

        r = random.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return n - 1  # fallback

class UCBPolicy(_BanditStatsBase):
    """
    UCB1-style policy over mean reward.
    """
    def __init__(self, c: float = 2.0):
        super().__init__()
        self.c = c

    def select(self, obs: List[List[float]]) -> int:
        n = len(obs)
        self._ensure_arms(n)

        # Try each arm at least once
        for i in range(n):
            if self.counts[i] == 0:
                return i

        total_counts = sum(self.counts[:n])
        ucb_vals = []
        for i in range(n):
            mean = self.means[i]
            bonus = self.c * math.sqrt(math.log(total_counts) / self.counts[i])
            ucb_vals.append(mean + bonus)

        return int(max(range(n), key=lambda i: ucb_vals[i]))


class ThompsonSamplingPolicy(_BanditStatsBase):
    """
    Thompson Sampling with Gaussian approximation over reward.
    """
    def __init__(self, prior_var: float = 1.0):
        super().__init__()
        self.prior_var = prior_var

    def select(self, obs: List[List[float]]) -> int:
        n = len(obs)
        self._ensure_arms(n)

        samples = []
        for i in range(n):
            c = self.counts[i]
            if c == 0:
                mean = 0.0
                var = self.prior_var * 10.0
            else:
                mean = self.means[i]
                # approximate posterior variance
                var = self._variance(i) / max(c, 1) + self.prior_var
            std = math.sqrt(max(var, 1e-8))
            samples.append(random.gauss(mean, std))

        return int(max(range(n), key=lambda i: samples[i]))


# ────────────────────────────────────────────────────────────
# PPO Policy (CPU) — on-policy updates from live traffic
# Thread-safe buffers to handle concurrent requests
# ────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.pi(h), self.v(h)


class PPOPolicy:
    def __init__(
        self,
        n_slots: int = 16,
        obs_dim: int = 6,         # default; will adjust dynamically
        lr: float = 3e-4,
        clip: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        update_every: int = 128,
        minibatch: int = 64,
        epochs: int = 4,
    ):
        # hyperparams
        self.n_slots = n_slots
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        # allow override via env (compose sets UPDATE_EVERY optionally)
        env_update = os.getenv("UPDATE_EVERY")
        self.update_every = int(env_update) if env_update else update_every
        self.minibatch = minibatch
        self.epochs = epochs

        # model/opt
        self.model = MLP(obs_dim, n_slots)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # online trajectory buffers
        self.obs_buf: List[List[float]] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.val_buf: List[float] = []
        self.rew_buf: List[float] = []

        # concurrency + persistence
        self._lock = threading.Lock()
        self._updating = False
        self._updates = 0
        self._ckpt_path = os.getenv("CHECKPOINT_PATH", "/app/checkpoints/ppo.pt")
        self._save_every = int(os.getenv("SAVE_EVERY_UPDATES", "0"))

        # optional Prometheus counter (safe if client not present)
        try:
            from prometheus_client import Counter
            self._ppo_updates_ctr = Counter("ppo_updates_total", "Total PPO parameter updates")
        except Exception:
            self._ppo_updates_ctr = None

    # ---------- helpers ----------
    def _mask_logits(self, logits: torch.Tensor, valid_n: int) -> torch.Tensor:
        if valid_n < logits.shape[-1]:
            mask = torch.full_like(logits, float('-inf'))
            mask[..., :valid_n] = 0.0
            logits = logits + mask
        return logits

    def _pool_obs(self, obs: List[List[float]]) -> torch.Tensor:
        """
        obs is per-backend rows:
          [mean, p95, queue, one_hot_step...]
        We want a single state vector:
          [min(mean,p95,queue), mean(mean,p95,queue)] + one_hot_step
        IMPORTANT: We do NOT pool the one-hot tag; we copy it through directly.
        """
        arr = torch.tensor(obs, dtype=torch.float32)  # (n_backends, obs_dim)
        base = arr[:, :3]  # mean, p95, queue
        mins = base.min(dim=0).values  # (3,)
        means = base.mean(dim=0)  # (3,)

        # Pool extra columns (one-hot + context) by mean so they pass through
        extras = arr[:, 3:].mean(dim=0) if arr.shape[1] > 3 else torch.empty(0)

        pooled = torch.cat([mins, means, extras])  # (6 + extras_dim,)
        return pooled.unsqueeze(0)  # (1, 6 + extras_dim)

    def _align_buffers_locked(self):
        mn = min(len(self.obs_buf), len(self.act_buf), len(self.logp_buf),
                 len(self.val_buf), len(self.rew_buf))
        self.obs_buf = self.obs_buf[:mn]
        self.act_buf = self.act_buf[:mn]
        self.logp_buf = self.logp_buf[:mn]
        self.val_buf = self.val_buf[:mn]
        self.rew_buf = self.rew_buf[:mn]
        return mn

    # ---------- API ----------
    def select(self, obs: List[List[float]], valid_n: int | None = None) -> int:
        valid_n = valid_n or len(obs)
        pooled = self._pool_obs(obs)

        # adapt model input dim if step-tag one-hot changed
        if self.obs_dim != pooled.shape[-1]:
            self.obs_dim = pooled.shape[-1]
            self.model = MLP(self.obs_dim, self.n_slots)
            # keep same LR as before
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.opt.param_groups[0]['lr'])

        logits, value = self.model(pooled)
        logits = self._mask_logits(logits, valid_n)
        dist = torch.distributions.Categorical(logits=logits)
        act_t = dist.sample()
        act = int(act_t.item())
        logp = float(dist.log_prob(act_t).item())
        val = float(value.squeeze().item())

        with self._lock:
            self.obs_buf.append(pooled.squeeze(0).tolist())
            self.act_buf.append(act)
            self.logp_buf.append(logp)
            self.val_buf.append(val)
        return act

    def observe(self, reward: float):
        with self._lock:
            self.rew_buf.append(float(reward))
            # Keep buffers aligned at all times
            n = self._align_buffers_locked()
            # Only one thread triggers an update
            if (not self._updating) and n >= self.update_every:
                self._updating = True
        # Do the update outside the lock (but guard re-entrancy)
        if self._updating:
            try:
                self._update()
            finally:
                with self._lock:
                    self._updating = False

    # ---------- training ----------
    def _update(self):
        # Snapshot under lock to avoid long-held lock while training
        with self._lock:
            n = self._align_buffers_locked()
            if n == 0:
                return
            obs = torch.tensor(self.obs_buf, dtype=torch.float32)
            acts = torch.tensor(self.act_buf, dtype=torch.int64)
            old_logp = torch.tensor(self.logp_buf, dtype=torch.float32)
            vals = torch.tensor(self.val_buf, dtype=torch.float32)
            rews = torch.tensor(self.rew_buf, dtype=torch.float32)
            # Clear now; if requests arrive during update, they accumulate for next round
            self.obs_buf.clear(); self.act_buf.clear(); self.logp_buf.clear(); self.val_buf.clear(); self.rew_buf.clear()

        # Single-step returns/advantages (request-level RL)
        adv = rews - vals
        ret = rews
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), self.minibatch):
                j = idx[start:start + self.minibatch]
                b_obs, b_acts = obs[j], acts[j]
                b_old_logp, b_adv, b_ret = old_logp[j], adv[j], ret[j]

                logits, v = self.model(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_acts)
                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(v.squeeze(-1), b_ret)
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

        # bookkeeping + persistence
        self._updates += 1
        if self._ppo_updates_ctr is not None:
            try:
                self._ppo_updates_ctr.inc()
            except Exception:
                pass
        if self._save_every > 0 and (self._updates % self._save_every == 0):
            try:
                self.save(self._ckpt_path)
                print(f"[policy] Saved checkpoint to {self._ckpt_path} (update {self._updates})")
            except Exception as e:
                print(f"[policy] Save failed: {e}")

    # ---------- persistence ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "obs_dim": self.obs_dim,
            "n_slots": self.n_slots,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.obs_dim = ckpt.get("obs_dim", self.obs_dim)
        self.n_slots = ckpt.get("n_slots", self.n_slots)
        self.model = MLP(self.obs_dim, self.n_slots)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.opt.param_groups[0]['lr'])
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])