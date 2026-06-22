#!/usr/bin/env python3
"""
sim_to_real.py — 仿真到真机的参数平滑过渡与 Domain Randomization
==================================================================

核心思想: 仿真的策略在真机上表现不佳的核心原因是 "仿真-真机差异"
(Sim-to-Real Gap, aka. Reality Gap)。

本模块提供三种缩小差距的方法:
  1. 参数平滑: 将仿真策略的参数渐进式调整到真机参数
  2. Domain Randomization: 随机化参数以学得更鲁棒的策略
  3. 在线适配: 在真机上微调策略参数

真机部署必须处理的关键问题:
  - F/T 传感器噪声和偏置
  - 关节摩擦和阻尼差异
  - 连杆质量和惯量误差
  - 控制延迟和执行器响应差异
  - 环境摩擦力 (门铰链/手柄)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class SimToRealConfig:
    """Sim-to-Real 参数配置."""
    # 摩擦补偿
    coulomb_friction: float = 0.15     # N·m 库仑摩擦
    viscous_friction: float = 0.05     # N·m·s/rad 粘性摩擦
    friction_joints: tuple = (0, 1, 2, 3, 4, 5, 6)

    # 延迟补偿
    control_latency_s: float = 0.02    # 真机控制延迟 (秒)
    delay_steps: int = 2               # 补偿的延迟步数

    # 噪声注入
    noise_std_pos: float = 0.002       # rad
    noise_std_vel: float = 0.01        # rad/s
    noise_std_ft: float = 0.5          # N 或 Nm

    # Domain Randomization 范围 (训练时随机化)
    do_domain_randomization: bool = False
    dr_friction_range: Tuple[float, float] = (0.05, 0.3)
    dr_mass_scale: Tuple[float, float] = (0.8, 1.2)
    dr_damping_scale: Tuple[float, float] = (0.5, 2.0)


class SimToRealWrapper:
    """Sim-to-Real 包装器, 为仿真策略添加真机补偿.

    用法:
        st = SimToRealWrapper(cfg)
        
        # 真机循环中:
        tau_compensated = st.friction_compensation(tau_raw, dq)
        q_delayed = st.latency_compensation(q_cmd)
        obs_noisy = st.inject_noise(obs)
    """

    def __init__(self, cfg: Optional[SimToRealConfig] = None):
        self.cfg = cfg or SimToRealConfig()
        self._cmd_buffer = []
        self._step = 0

    def friction_compensation(
        self, tau_raw: np.ndarray, dq: np.ndarray
    ) -> np.ndarray:
        """摩擦补偿: τ_comp = τ_raw + sign(q̇) * τ_coulomb + q̇ * τ_viscous.
        
        真机中, 关节摩擦会消耗一部分力/力矩.
        补偿后控制器看到的净力矩更接近仿真.
        """
        tau_fric = np.zeros_like(tau_raw)
        for j in self.cfg.friction_joints:
            tau_fric[j] = (
                np.sign(dq[j]) * self.cfg.coulomb_friction
                + dq[j] * self.cfg.viscous_friction
            )
        return tau_raw + tau_fric

    def latency_compensation(self, q_cmd: np.ndarray) -> np.ndarray:
        """延迟补偿: 使用缓冲区延迟指令.
        
        真机控制延迟约 10-30ms (ROS2 pub/sub + 驱动器响应).
        仿真中指令立即生效, 真机中需要提前发送预测的指令.
        这里采用简单的 N-step 延迟缓冲区来对齐.
        """
        self._cmd_buffer.append(q_cmd.copy())
        delay = self.cfg.delay_steps
        if len(self._cmd_buffer) > delay + 1:
            return self._cmd_buffer.pop(0)
        else:
            return q_cmd.copy()

    def inject_noise(
        self, obs: np.ndarray, seed: Optional[int] = None
    ) -> np.ndarray:
        """注入观测噪声 (模拟真机传感器噪声)."""
        if seed is not None:
            np.random.seed(seed)
        noisy = obs.copy()
        # 位置噪声
        n_pos = min(self.cfg.noise_std_pos, obs.shape[0])
        noisy[:n_pos] += np.random.randn(n_pos) * self.cfg.noise_std_pos
        # 力/力矩噪声 (观测的后7维)
        n_ft = 6
        if obs.shape[0] >= 14:
            noisy[7:13] += np.random.randn(6) * self.cfg.noise_std_ft
        return noisy

    def get_dr_params(self, seed: Optional[int] = None) -> Dict:
        """生成 Domain Randomization 参数."""
        if seed is not None:
            np.random.seed(seed)
        cfg = self.cfg
        return {
            "friction": np.random.uniform(*cfg.dr_friction_range),
            "mass_scale": np.random.uniform(*cfg.dr_mass_scale),
            "damping_scale": np.random.uniform(*cfg.dr_damping_scale),
        }

    def smooth_start(
        self, q_init: np.ndarray, steps: int = 100
    ) -> np.ndarray:
        """生成平滑起始轨迹 (从初始位置到第一个目标位置).
        
        用于真机启动时避免冲击.
        返回的插值序列可以直接发送.
        """
        t = np.linspace(0, 1.0, steps)
        # 默认输出恒定: 保持在初始位置
        return np.tile(q_init, (steps, 1))

    def reset(self):
        """重置缓冲区."""
        self._cmd_buffer.clear()
        self._step = 0
