# =============================
# File: router/policy.py
# =============================
from __future__ import annotations

import random
import threading
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# Simple epsilon-greedy baseline (kept for ablations)
# ────────────────────────────────────────────────────────────
class EpsilonGreedyPolicy:
    """Selects backend with lowest latency proxy with epsilon exploration."""
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.buffer = []

    def select(self, obs: List[List[float]]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(obs))
        # score = mean + 0.5 * p95 + 0.1 * queue
        scores = [o[0] + 0.5 * o[1] + 0.1 * o[2] for o in obs]
        return int(min(range(len(scores)), key=lambda i: scores[i]))

    def observe(self, o, a, r):
        self.buffer.append((o, a, r))
        if len(self.buffer) > 10_000:
            self.buffer = self.buffer[-5_000:]


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
        obs_dim: int = 6,         # pooled features size (min/mean for mean,p95,queue)
        lr: float = 3e-4,
        clip: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        update_every: int = 512,
        minibatch: int = 128,
        epochs: int = 4,
    ):
        self.n_slots = n_slots
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.update_every = update_every
        self.minibatch = minibatch
        self.epochs = epochs

        self.model = MLP(obs_dim, n_slots)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Trajectory buffers
        self.obs_buf: List[List[float]] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.val_buf: List[float] = []
        self.rew_buf: List[float] = []

        # Concurrency control
        self._lock = threading.Lock()
        self._updating = False   # prevent re-entrant/parallel updates

    # ---------- helpers ----------
    def _mask_logits(self, logits: torch.Tensor, valid_n: int) -> torch.Tensor:
        if valid_n < logits.shape[-1]:
            mask = torch.full_like(logits, float('-inf'))
            mask[..., :valid_n] = 0.0
            logits = logits + mask
        return logits

    def _pool_obs(self, obs: List[List[float]]) -> torch.Tensor:
        # obs[i] = [mean, p95, q]
        arr = torch.tensor(obs, dtype=torch.float32)
        pooled = torch.stack([
            arr[:, 0].min(),  arr[:, 1].min(),  arr[:, 2].min(),
            arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(),
        ])
        return pooled.unsqueeze(0)  # (1, 6)

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

        # Ensure model input dim matches pooled size
        if self.obs_dim != pooled.shape[-1]:
            self.obs_dim = pooled.shape[-1]
            self.model = MLP(self.obs_dim, self.n_slots)
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
