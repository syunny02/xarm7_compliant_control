#!/usr/bin/env python3
"""
policy.py — 策略加载与推理
===========================
与 ROS2 xarm7_door_policy/policy_loader.py 完全兼容。

支持的文件格式 (torch.save):
    torch.save({
        'arch': 'mlp',
        'obs_dim': 61,
        'act_dim': 8,
        'hidden': [512, 256, 256],
        'state_dict': actor.state_dict(),
        'obs_mean': np.ndarray(obs_dim,),   # RunningMeanStd mean
        'obs_var': np.ndarray(obs_dim,),    # RunningMeanStd var
        'act_low': np.ndarray(act_dim,),    # action denorm lower bound
        'act_high': np.ndarray(act_dim,),   # action denorm upper bound
    }, 'policy.pt')

用法:
    policy = LoadedPolicy('policy.pt', device='cuda')
    action = policy.infer(obs)          # shape (act_dim,)
    norm_action = policy.infer_normalized(obs)  # [-1, 1] raw
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class MLPActor(nn.Module):
    """多层感知机 Actor 网络 (Tanh 激活)."""
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers += [nn.Linear(last, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoadedPolicy:
    """加载训练好的 .pt 策略文件并运行推理.

    属性:
        obs_dim: 观测维度
        act_dim: 动作维度 (7 关节 + 1 夹爪 = 8)
        device: 推理设备
    """

    def __init__(self, path: str, device: str = 'cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.arch = ckpt.get('arch', 'mlp')
        self.obs_dim = int(ckpt['obs_dim'])
        self.act_dim = int(ckpt['act_dim'])
        hidden = tuple(ckpt.get('hidden', (256, 256)))

        if self.arch == 'mlp':
            self.model = MLPActor(self.obs_dim, self.act_dim, hidden)
        else:
            raise NotImplementedError(f"Unsupported arch: {self.arch}")
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(device).eval()

        self.obs_mean = np.asarray(
            ckpt.get('obs_mean', np.zeros(self.obs_dim)), dtype=np.float32
        )
        self.obs_var = np.asarray(
            ckpt.get('obs_var', np.ones(self.obs_dim)), dtype=np.float32
        )
        self.act_low = np.asarray(
            ckpt.get('act_low', -np.ones(self.act_dim)), dtype=np.float32
        )
        self.act_high = np.asarray(
            ckpt.get('act_high', np.ones(self.act_dim)), dtype=np.float32
        )
        self.device = device

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """观测归一化: (obs - mean) / std."""
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)

    def denorm_action(self, a: np.ndarray) -> np.ndarray:
        """将 [-1, 1] 归一化动作映射到实际范围."""
        return self.act_low + 0.5 * (a + 1.0) * (self.act_high - self.act_low)

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """策略推理 (返回去归一化的实际动作).

        Args:
            obs: 原始观测向量 (obs_dim,)

        Returns:
            action: 去归一化动作 (act_dim,)
        """
        assert obs.shape[0] == self.obs_dim, \
            f"obs_dim mismatch: {obs.shape[0]} != {self.obs_dim}"
        x = self.normalize(obs.astype(np.float32))
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        a = self.model(t).cpu().numpy()[0]  # (act_dim,) in [-1, 1]
        return self.denorm_action(a)

    @torch.no_grad()
    def infer_normalized(self, obs: np.ndarray) -> np.ndarray:
        """策略推理 (返回 [-1, 1] 原始动作)."""
        assert obs.shape[0] == self.obs_dim, \
            f"obs_dim mismatch: {obs.shape[0]} != {self.obs_dim}"
        x = self.normalize(obs.astype(np.float32))
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return self.model(t).cpu().numpy()[0]

    def save_export(self, path: str):
        """导出为 ONNX 格式 (真机部署优化用)."""
        dummy = torch.randn(1, self.obs_dim, device=self.device)
        torch.onnx.export(self.model, dummy, path,
                          input_names=['obs'],
                          output_names=['action'],
                          dynamic_axes={'obs': {0: 'batch'},
                                        'action': {0: 'batch'}})
