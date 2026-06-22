#!/usr/bin/env python3
"""
admittance.py — 关节空间导纳滤波器
====================================
理论:  M · Δq̈ + D · Δq̇ + K · Δq = τ_ext
其中:
  M: 虚拟质量 (惯性)
  D: 虚拟阻尼
  K: 虚拟刚度
  τ_ext: 外部关节力矩 (来自 F/T 传感器 × Jacobian 转置)

积分方法: 半隐式 Euler (比显式更稳定)

真机部署要点:
  - 每个关节独立的解耦导纳 (假设关节间耦合较小)
  - 变量刚度: 高接触力 → 低刚度 (柔性), 低接触力 → 高刚度 (精确)
  - 抗饱和 (anti-windup): 位置偏移超限时减速
  - 安全限位: max_delta_rad 硬限制

参考:
  - Hogan, N. (1985). Impedance Control: An Approach to Manipulation.
  - Ott, C. (2008). Cartesian Impedance Control of Redundant Robots.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AdmittanceConfig:
    """导纳滤波器参数 (所有关节共享标量)."""
    mass: float = 1.0          # M  — 虚拟质量 (kg)
    damping: float = 12.0      # D  — 虚拟阻尼 (N·s/m)
    stiffness: float = 40.0    # K  — 虚拟刚度 (N/m)
    max_delta_rad: float = 0.15   # 最大位置偏移 (安全限位, rad)
    dt: float = 0.01           # 积分步长 (秒)

    # 变量刚度 (可选)
    var_stiffness: bool = False
    K_min: float = 5.0
    K_max: float = 80.0
    force_threshold: float = 5.0   # N — 低于此阈值使用 K_max (刚性);
                                   # 高于此阈值逐渐过渡到 K_min (柔性)

    # 策略刚度覆盖 (来自 Cartesian VIC 策略输出)
    stiffness_from_policy: bool = False  # True 时允许外部覆盖 stiffness

    # 关节单独缩放系数 (真机调参用)
    joint_scales: tuple = field(default_factory=lambda: (1.0,) * 7)


class AdmittanceFilter:
    """关节空间解耦导纳控制器.

    用法:
        adm = AdmittanceFilter(ndof=7, cfg=AdmittanceConfig())
        adm.reset(q_init)
        q_cmd = adm.update(q_desired, tau_external)

    真机循环 (50-100Hz):
        1. 读取 F/T 传感器 → wrench
        2. Jacobian 转置 → τ_ext = J(q)^T · wrench
        3. 导纳滤波 → Δq
        4. q_cmd = q_des + Δq
    """

    def __init__(self, ndof: int, cfg: Optional[AdmittanceConfig] = None):
        self.n = ndof
        self.cfg = cfg or AdmittanceConfig()
        self.dq = np.zeros(ndof)      # Δq  — 位置偏移
        self.dqd = np.zeros(ndof)     # Δq̇ — 速度偏移
        self.q_des = np.zeros(ndof)   # 上次期望位置 (来自策略)
        self._reset_filter = True

    def reset(self, q_init: np.ndarray):
        """重置导纳状态为初始位置."""
        self.dq[:] = 0.0
        self.dqd[:] = 0.0
        self.q_des = q_init.copy()
        self._reset_filter = True

    def set_stiffness_from_force(self, tau_mag: float):
        """根据接触力调整刚度.
        
        高接触力 → 低刚度 (更顺从)
        低接触力 → 高刚度 (更精确)
        """
        if not self.cfg.var_stiffness:
            return
        ft = self.cfg.force_threshold
        if tau_mag < ft:
            self.cfg.stiffness = self.cfg.K_max
        else:
            ratio = min(1.0, (tau_mag - ft) / 20.0)
            self.cfg.stiffness = self.cfg.K_min + ratio * (self.cfg.K_max - self.cfg.K_min)
            self.cfg.stiffness = np.clip(
                self.cfg.stiffness, self.cfg.K_min, self.cfg.K_max
            )

    def update(self, q_des: np.ndarray, tau_ext: np.ndarray) -> np.ndarray:
        """计算导纳修正后的关节指令.

        Args:
            q_des:  策略输出的期望关节位置 (ndof,)
            tau_ext: 外部关节力矩 (ndof,), 来自 J^T · wrench

        Returns:
            q_cmd: 柔性关节指令 = q_des + Δq (ndof,)
        """
        assert q_des.shape == (self.n,) and tau_ext.shape == (self.n,)

        # 首次调用或 des 突变时，直接输出并初始化
        if self._reset_filter:
            self.q_des = q_des.copy()
            self._reset_filter = False
            return q_des.copy()

        M = max(self.cfg.mass, 1e-6)
        D = self.cfg.damping
        K = self.cfg.stiffness
        dt = self.cfg.dt
        s = np.asarray(self.cfg.joint_scales, dtype=np.float64)

        # 半隐式 Euler 积分:
        #   ddq = (τ_ext - D·q̇ - K·q) / M
        #   q̇_{t+1} = q̇_t + ddq·dt  
        #   q_{t+1}  = q_t  + q̇_{t+1}·dt
        ddq = (s * tau_ext - D * self.dqd - K * self.dq) / M
        self.dqd += ddq * dt
        self.dq += self.dqd * dt

        # 安全限位
        np.clip(self.dq, -self.cfg.max_delta_rad, self.cfg.max_delta_rad, out=self.dq)

        # 抗饱和: 如果位置偏移达到限位, 速度衰减
        clipped = np.abs(self.dq) >= self.cfg.max_delta_rad * 0.99
        self.dqd[clipped] *= 0.5

        q_cmd = q_des + self.dq
        self.q_des = q_des.copy()
        return q_cmd

    def set_stiffness(self, K: float):
        """外部设置导纳刚度 (用于策略刚度覆盖).

        Args:
            K: 刚度值 (N/m), 会被 clamp 到 [1, 1000]
        """
        if self.cfg.stiffness_from_policy:
            self.cfg.stiffness = float(np.clip(K, 1.0, 1000.0))

    def get_offset(self) -> np.ndarray:
        """返回当前导纳偏移量 Δq."""
        return self.dq.copy()

    def get_velocity(self) -> np.ndarray:
        """返回当前导纳速度 Δq̇."""
        return self.dqd.copy()


class VariableAdmittance:
    """变量导纳控制器: 根据任务相位调整参数.

    开门任务相位:
      phase 0: 接近门 (高刚度, 精确跟踪)
      phase 1: 抓握手柄 (中等刚度, 允许顺应性调整)
      phase 2: 拉门 (低刚度, 高柔性以跟随手柄轨迹)
      phase 3: 开门完成 (恢复高刚度)
    """

    PHASE_PARAMS = {
        0: {"mass": 0.5, "damping": 20.0, "stiffness": 80.0, "max_delta": 0.03},
        1: {"mass": 1.0, "damping": 15.0, "stiffness": 40.0, "max_delta": 0.08},
        2: {"mass": 1.5, "damping": 10.0, "stiffness": 15.0, "max_delta": 0.15},
        3: {"mass": 0.5, "damping": 20.0, "stiffness": 60.0, "max_delta": 0.05},
    }

    def __init__(self, ndof: int, dt: float = 0.01):
        self.ndof = ndof
        self.dt = dt
        self._current_phase = 0
        self._filter = AdmittanceFilter(ndof, self._make_config(0, dt))

    def _make_config(self, phase: int, dt: float) -> AdmittanceConfig:
        p = self.PHASE_PARAMS.get(phase, self.PHASE_PARAMS[0])
        return AdmittanceConfig(
            mass=p["mass"], damping=p["damping"], stiffness=p["stiffness"],
            max_delta_rad=p["max_delta"], dt=dt,
        )

    def set_phase(self, phase: int):
        """切换任务相位，更新导纳参数."""
        if phase != self._current_phase:
            self._current_phase = phase
            old_dq = self._filter.dq.copy()
            old_dqd = self._filter.dqd.copy()
            self._filter = AdmittanceFilter(self.ndof, self._make_config(phase, self.dt))
            self._filter.dq = old_dq
            self._filter.dqd = old_dqd
            self._filter._reset_filter = False

    def set_stiffness(self, K: float):
        """外部设置导纳刚度 (透传给内部 filter)."""
        self._filter.set_stiffness(K)

    def reset(self, q_init: np.ndarray):
        self._filter.reset(q_init)

    def update(self, q_des: np.ndarray, tau_ext: np.ndarray) -> np.ndarray:
        return self._filter.update(q_des, tau_ext)
