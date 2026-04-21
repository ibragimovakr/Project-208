from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = type(model)(**{}) if False else None
        self.shadow = self._clone(model)

    def _clone(self, model: nn.Module) -> nn.Module:
        import copy
        shadow = copy.deepcopy(model).eval()
        for p in shadow.parameters():
            p.requires_grad = False
        return shadow

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))


def get_lambda_drift(epoch: int, cfg) -> float:
    if epoch <= cfg.warmup_epochs:
        return 0.0
    ramp_start = cfg.warmup_epochs + 1
    ramp_end = cfg.warmup_epochs + cfg.drift_ramp_epochs
    if epoch <= ramp_end:
        progress = (epoch - ramp_start + 1) / max(1, cfg.drift_ramp_epochs)
        return cfg.drift_loss_weight + progress * (cfg.drift_loss_weight_max - cfg.drift_loss_weight)
    return cfg.drift_loss_weight_max


def get_lambda_same_neg(epoch: int, cfg) -> float:
    if epoch <= cfg.warmup_epochs:
        return cfg.lambda_same_neg_start
    ramp_end = cfg.warmup_epochs + cfg.drift_ramp_epochs
    if epoch <= ramp_end:
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.drift_ramp_epochs)
        return cfg.lambda_same_neg_start + progress * (cfg.lambda_same_neg_end - cfg.lambda_same_neg_start)
    return cfg.lambda_same_neg_end


def subsample_spatial_positions(x: torch.Tensor, max_positions: int):
    t, c, h, w = x.shape
    l = h * w
    x = x.view(t, c, l)
    if l <= max_positions:
        return x, None
    idx = torch.randperm(l, device=x.device)[:max_positions]
    return x[:, :, idx], idx


def compute_feature_scale_from_banks(g_feats: torch.Tensor, p_feats: torch.Tensor, eps: float = 1e-6):
    c = g_feats.shape[1]
    g_bank = g_feats.permute(0, 2, 1).reshape(-1, c)
    p_bank = p_feats.permute(0, 2, 1).reshape(-1, c)
    if g_bank.numel() == 0 or p_bank.numel() == 0:
        return g_feats.new_tensor(1.0)
    d = torch.cdist(g_bank, p_bank, p=2)
    scale = d.mean() / math.sqrt(c)
    return scale.detach().clamp_min(eps)


def compute_simple_conditional_drift(x: torch.Tensor, y_pos: torch.Tensor, tau: float, lambda_pos: float, lambda_same_neg: float):
    dist_pos = torch.cdist(x, y_pos, p=2)
    logit_pos = -dist_pos / tau
    w_pos = torch.softmax(logit_pos, dim=1)
    mu_pos = w_pos @ y_pos
    v_pos = mu_pos - x

    if x.shape[0] > 1:
        dist_same = torch.cdist(x, x, p=2)
        dist_same = dist_same + torch.eye(x.shape[0], device=x.device, dtype=x.dtype) * 1e6
        logit_same = -dist_same / tau
        w_same = torch.softmax(logit_same, dim=1)
        mu_same = w_same @ x
        v_same = mu_same - x
        a_same_mean = float(w_same.mean().detach().cpu())
    else:
        v_same = torch.zeros_like(x)
        a_same_mean = 0.0

    v = lambda_pos * v_pos - lambda_same_neg * v_same
    info = {
        "A_pos_mean": float(w_pos.mean().detach().cpu()),
        "A_same_mean": a_same_mean,
        "V_rms_before_norm": float(torch.sqrt(torch.mean(v ** 2) + 1e-8).detach().cpu()),
    }
    return v, info


class SingleLevelConditionalDriftingLoss(nn.Module):
    def __init__(self, encoder: nn.Module, cfg):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg

    def forward(self, x_gen: torch.Tensor, x_pos: torch.Tensor, x_up: torch.Tensor, epoch: int):
        B, N, C, H, W = x_gen.shape
        M = x_pos.shape[1]
        lambda_same_neg = get_lambda_same_neg(epoch, self.cfg)

        if self.cfg.drift_fp32:
            x_gen = x_gen.float()
            x_pos = x_pos.float()
            x_up = x_up.float()

        x_gen_flat = x_gen.reshape(B * N, C, H, W)
        x_pos_flat = x_pos.reshape(B * M, C, H, W)
        x_up_flat_for_gen = x_up[:, None].expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
        x_up_flat_for_pos = x_up[:, None].expand(-1, M, -1, -1, -1).reshape(B * M, C, H, W)

        gen_maps = self.encoder(x_gen_flat)
        pos_maps = self.encoder(x_pos_flat)
        up_maps_gen = self.encoder(x_up_flat_for_gen)
        up_maps_pos = self.encoder(x_up_flat_for_pos)

        s = self.cfg.feature_scale_index
        g_map = gen_maps[s] - up_maps_gen[s]
        p_map = pos_maps[s] - up_maps_pos[s]

        _, Cg, Hs, Ws = g_map.shape
        g_map = g_map.view(B, N, Cg, Hs, Ws)
        p_map = p_map.view(B, M, Cg, Hs, Ws)

        g_list, p_list = [], []
        for b in range(B):
            g_b, idx = subsample_spatial_positions(g_map[b], self.cfg.feature_max_positions)
            p_full = p_map[b].view(M, Cg, Hs * Ws)
            p_b = p_full if idx is None else p_full[:, :, idx]
            g_list.append(g_b)
            p_list.append(p_b)

        g_batch = torch.cat(g_list, dim=2)
        p_batch = torch.cat(p_list, dim=2)
        feat_scale = compute_feature_scale_from_banks(g_batch, p_batch, eps=self.cfg.norm_eps).clamp_min(self.cfg.feat_scale_min)
        g_list = [g_b / feat_scale for g_b in g_list]
        p_list = [p_b / feat_scale for p_b in p_list]

        raw_vs_per_b = []
        all_raw_vs = []
        logs_a_pos, logs_a_same, logs_v_rms = [], [], []
        for b in range(B):
            g_b = g_list[b]
            p_b = p_list[b]
            L = g_b.shape[-1]
            raw_vs_b = []
            for l in range(L):
                x_loc = g_b[:, :, l]
                p_loc = p_b[:, :, l]
                v_loc, info = compute_simple_conditional_drift(
                    x=x_loc,
                    y_pos=p_loc,
                    tau=self.cfg.temperature,
                    lambda_pos=self.cfg.lambda_pos,
                    lambda_same_neg=lambda_same_neg,
                )
                raw_vs_b.append(v_loc)
                all_raw_vs.append(v_loc)
                logs_a_pos.append(info["A_pos_mean"])
                logs_a_same.append(info["A_same_mean"])
                logs_v_rms.append(info["V_rms_before_norm"])
            raw_vs_per_b.append(raw_vs_b)

        if len(all_raw_vs) > 0:
            all_raw_vs = torch.stack(all_raw_vs, dim=0)
            drift_scale = torch.sqrt(torch.mean(all_raw_vs ** 2) + self.cfg.norm_eps).detach()
        else:
            drift_scale = x_gen.new_tensor(1.0)

        drift_norm_factor = drift_scale.clamp(min=self.cfg.drift_scale_center, max=self.cfg.drift_scale_max)

        total_loss = x_gen.new_tensor(0.0)
        num_items = 0
        for b in range(B):
            g_b = g_list[b]
            raw_vs_b = raw_vs_per_b[b]
            L = g_b.shape[-1]
            for l in range(L):
                x_loc = g_b[:, :, l]
                v_loc = raw_vs_b[l] / drift_norm_factor
                target = (x_loc + self.cfg.drift_step_size * v_loc).detach()
                total_loss = total_loss + F.mse_loss(x_loc, target)
                num_items += 1

        total_loss = total_loss / max(1, num_items)
        info = {
            "lambda_same_neg": float(lambda_same_neg),
            "scale": s,
            "A_pos_mean": float(np.mean(logs_a_pos)) if logs_a_pos else float("nan"),
            "A_same_mean": float(np.mean(logs_a_same)) if logs_a_same else float("nan"),
            "V_rms": float(np.mean(logs_v_rms)) if logs_v_rms else float("nan"),
            "feat_scale": float(feat_scale.detach().cpu()),
            "drift_scale": float(drift_scale.cpu()),
            "drift_norm_factor": float(drift_norm_factor.cpu()),
        }
        return total_loss, info
