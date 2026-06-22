#!/usr/bin/env python3
"""
controller.py — 柔性控制主流水线
================================
将以下模块串联为完整的控制循环:

  观测 → 策略推理 → 导纳滤波 → 安全监控 → 关节指令

                                 ┌─────────────┐
      ┌───┐    ┌──────┐    ┌───┴──┐    ┌─────┴───┐
      │MuJoCo├──►Policy├──►Admit.├──►Safety ├──►Joint Cmd
      │/ROS │    │      │    │Filter│    │Monitor │
      └───┘    └──────┘    └───┬──┘    └─────┬───┘
                               │              │
                          ┌────▼────┐    ┌────▼────┐
                          │ Wrench  │    │ Sim2Real│
                          │ Mapper  │    │ Wrapper │
                          └─────────┘    └─────────┘

真机循环 (50-100 Hz):
  while running:
    - 读关节位置 (q) 和速度 (dq)
    - 读 F/T 传感器 (wrench_raw)
    - 计算 τ_ext = J^T · wrench (含滤波/补偿)
    - 策略推理 → q_des (PPO/RL 策略)
    - 导纳滤波 → q_cmd = q_des + Δq (含限位)
    - 安全检查: 力/力矩/位置/速度
    - 发送关节指令
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from .core.admittance import AdmittanceFilter, AdmittanceConfig, VariableAdmittance
from .core.policy import LoadedPolicy
from .core.wrench import WrenchMapper, WrenchConfig


@dataclass
class SafetyLimits:
    """安全监控参数."""
    max_joint_pos: tuple = (2.97, 2.97, 2.97, 2.97, 2.97, 2.97, 2.97)
    min_joint_pos: tuple = (-2.97,) * 7
    max_joint_vel: float = 1.5       # rad/s
    max_command_delta: float = 0.05  # rad (单步最大变化)
    max_force_n: float = 150.0       # N (总力)
    max_torque_nm: float = 30.0      # Nm (总力矩)
    emergency_stop_tau: float = 50.0 # N·m (紧急停止阈值)

    @property
    def pos_min(self) -> np.ndarray:
        return np.array(self.min_joint_pos)

    @property
    def pos_max(self) -> np.ndarray:
        return np.array(self.max_joint_pos)


@dataclass
class CompliantControllerConfig:
    """柔性控制器完整配置."""
    ndof: int = 7                    # 关节数 (xArm7)
    policy_path: str = ""            # 策略文件路径
    device: str = "cpu"              # 推理设备
    control_dt: float = 0.01         # 控制周期 (秒, 100Hz)
    use_mujoco: bool = True          # 仿真模式 / 真机模式
    scene_xml: str = ""              # MuJoCo XML 路径 (仿真模式)
    tcp_body_name: str = "link_tcp"  # 末端坐标系名 (Jacobian 计算点)

    # --- 笛卡尔 VIC 模式 ---
    cartesian_vic: bool = False      # True: step() 接受 vic_action=[dx,K,gripper]
    vic_act_dim: int = 8             # [dx(3), drot(3), K(1), gripper(1)]
    stiffness_from_policy: bool = True  # 用策略输出的 K 覆盖导纳刚度
    damping_ratio: float = 1.0       # 临界阻尼比 (用于 impedance: D=2·√(K·m))
    eff_mass: float = 1.0            # 有效质量 (kg), 同训练 env 的 EFF_MASS_DEFAULT

    # 子模块配置
    admittance: AdmittanceConfig = field(default_factory=AdmittanceConfig)
    wrench: WrenchConfig = field(default_factory=WrenchConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    # 变量刚度
    variable_admittance: bool = False
    task_phase: int = 0


class CompliantController:
    """柔性控制器主流水线.

    两种运行模式:
      1. 仿真 (use_mujoco=True) — 在 MuJoCo 中验证
      2. 真机 (use_mujoco=False) — 实际机器人部署

    用法:
        ctrl = CompliantController(cfg)
        ctrl.reset(q_init)
        while running:
            q_cmd = ctrl.step(q_current, wrench_raw, obs)

    注意:
        MuJoCo 模式下 wrench_raw 由仿真碰撞力计算;
        真机模式下由 F/T 传感器读取.
    """

    def __init__(self, cfg: CompliantControllerConfig):
        self.cfg = cfg
        self.dt = cfg.control_dt
        self._mujoco = None
        self._mujoco_data = None
        self._jacobian_provider = None

        # 1) 加载策略
        self.policy: Optional[LoadedPolicy] = None
        if cfg.policy_path and Path(cfg.policy_path).exists():
            self.policy = LoadedPolicy(cfg.policy_path, device=cfg.device)
            print(f"[Controller] Loaded policy: {cfg.policy_path}")

        # 2) 导纳滤波器
        if cfg.variable_admittance:
            self.admittance = VariableAdmittance(cfg.ndof, dt=cfg.control_dt)
        else:
            self.admittance = AdmittanceFilter(cfg.ndof, cfg.admittance)

        # 3) 力/力矩映射器
        if cfg.use_mujoco:
            self._init_mujoco()
            self.wrench = WrenchMapper(
                model=self._mujoco,
                data=self._mujoco_data,
                cfg=cfg.wrench,
                body_name=cfg.tcp_body_name,
            )
        else:
            self.wrench = WrenchMapper(cfg=cfg.wrench)

        # 4) 安全监控器
        self.safety = cfg.safety

        # 5) Jacobian 提供器 (笛卡尔模式用)
        if cfg.cartesian_vic and cfg.use_mujoco:
            from .core.jacobian_provider import MuJoCoJacobianProvider
            self._jacobian_provider = MuJoCoJacobianProvider(
                self._mujoco, self._mujoco_data, cfg.tcp_body_name, cfg.ndof,
            )
        elif cfg.cartesian_vic:
            from .core.jacobian_provider import ApproxJacobianProvider
            self._jacobian_provider = ApproxJacobianProvider(cfg.ndof)

        # 状态
        self.q_last = np.zeros(cfg.ndof)
        self.tau_ext_last = np.zeros(cfg.ndof)
        self.delta_cmd = np.zeros(cfg.ndof)
        self.step_count = 0
        self._last_gripper = 0.0

    def _init_mujoco(self):
        """初始化 MuJoCo 仿真环境."""
        import mujoco
        xml = self.cfg.scene_xml
        if xml and Path(xml).exists():
            self._mujoco = mujoco.MjModel.from_xml_path(xml)
        else:
            # 默认场景 (可用于快速测试)
            default_xml = """<?xml version="1.0"?>
<mujoco model="xarm7_test">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link0" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-2.97 2.97"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint2" type="hinge" axis="0 1 0" limited="true" range="-2.97 2.97"/>
        <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.25" rgba="0.2 0.6 0.8 1"/>
        <body name="link2" pos="0 0 0.25">
          <joint name="joint3" type="hinge" axis="0 1 0" limited="true" range="-2.97 2.97"/>
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.25" rgba="0.2 0.6 0.8 1"/>
          <body name="link3" pos="0 0 0.25">
            <joint name="joint4" type="hinge" axis="0 1 0" limited="true" range="-2.97 2.97"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.2" rgba="0.2 0.6 0.8 1"/>
            <body name="link4" pos="0 0 0.2">
              <joint name="joint5" type="hinge" axis="0 1 0" limited="true" range="-2.97 2.97"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.2 0.6 0.8 1"/>
              <body name="link5" pos="0 0 0.2">
                <joint name="joint6" type="hinge" axis="0 1 0" limited="true" range="-2.97 2.97"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.15" rgba="0.2 0.6 0.8 1"/>
                <body name="link6" pos="0 0 0.15">
                  <joint name="joint7" type="hinge" axis="1 0 0" limited="true" range="-2.97 2.97"/>
                  <geom type="sphere" size="0.04" rgba="0.9 0.2 0.2 1"/>
                  <body name="link7" pos="0 0 0">
                    <geom type="box" size="0.02 0.02 0.02" rgba="0.9 0.9 0.2 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
            import tempfile
            f = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w')
            f.write(default_xml)
            f.close()
            self._mujoco = mujoco.MjModel.from_xml_path(f.name)
        self._mujoco_data = mujoco.MjData(self._mujoco)

    def reset(self, q_init: np.ndarray):
        """重置控制器状态."""
        self.admittance.reset(q_init)
        self.q_last = q_init.copy()
        self.tau_ext_last[:] = 0.0
        self.delta_cmd[:] = 0.0
        self.step_count = 0
        self._last_gripper = 0.0
        if self._mujoco_data is not None:
            self._mujoco_data.qpos[:self.cfg.ndof] = q_init
            import mujoco
            mujoco.mj_forward(self._mujoco, self._mujoco_data)

    def _get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """获取当前关节位置的末端 Jacobian (6×ndof).

        优先使用 MuJoCo 精确 Jacobian，fallback 到近似 Jacobian.
        """
        if self._jacobian_provider is not None:
            return self._jacobian_provider.get_jacobian(q)

        # Fallback: 用 wrench mapper 内置的 MuJoCo Jacobian
        if hasattr(self.wrench, '_jac') and self.wrench._jac is not None:
            ndof = self.cfg.ndof
            import mujoco
            self._mujoco_data.qpos[:ndof] = q[:ndof]
            mujoco.mj_forward(self._mujoco, self._mujoco_data)
            mujoco.mj_jacBody(
                self._mujoco, self._mujoco_data,
                self.wrench._jac[:3], self.wrench._jac[3:],
                self.wrench._body_id,
            )
            return self.wrench._jac[:, :ndof].copy()

        # 最后 fallback: 单位矩阵 (仅用于调试)
        return np.eye(6, self.cfg.ndof)

    def step(
        self,
        q: np.ndarray,
        wrench_raw: np.ndarray,
        obs: Optional[np.ndarray] = None,
        vic_action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """执行一个控制步.

        两种模式:
          1. 关节模式 (默认): obs → policy.infer → q_policy → admittance
          2. 笛卡尔 VIC 模式 (vic_action 非空):
             vic_action=[dx,dy,dz, drotx,droty,drotz, K, gripper]
             内部做差分IK (dq = J_pinv @ dx_cart) → q_des → admittance

        Args:
            q: 当前关节位置 (ndof,)
            wrench_raw: 原始六维力/力矩 (6,)
            obs: 完整观测向量 (obs_dim,) — 用于策略推理 (关节模式)
            vic_action: 笛卡尔 VIC 动作 (8,) — [dx(3), drot(3), K(1), gripper(1)]

        Returns:
            q_cmd: 最终关节指令 (ndof,)
        """
        # ---- 1. 策略推理 / 笛卡尔IK ----
        if vic_action is not None and self.cfg.cartesian_vic:
            # ---- 笛卡尔 VIC 模式 ----
            dx_cart = vic_action[:6].copy()   # [dx,dy,dz, drotx,droty,drotz]
            K_policy = float(vic_action[6])   # 策略输出的刚度
            self._last_gripper = float(vic_action[7])  # 夹爪指令，由外部使用

            # 差分IK: dq = J_pinv @ dx_cart
            J = self._get_jacobian(q)          # (6, ndof)
            J_pinv = np.linalg.pinv(J)         # (ndof, 6)
            dq = J_pinv @ dx_cart
            q_policy = q + dq

            # 策略刚度 → 导纳刚度
            if self.cfg.stiffness_from_policy:
                self.admittance.set_stiffness(K_policy)
        elif self.policy is not None and obs is not None:
            # ---- 关节模式: 策略推理 ----
            action = self.policy.infer(obs)
            q_policy = action[:self.cfg.ndof]
        else:
            # ---- 无策略: 位置保持 ----
            q_policy = q.copy()

        # ---- 2. 力/力矩映射 ----
        tau_ext = self.wrench.compute_tau(q, wrench_raw)
        self.tau_ext_last = tau_ext.copy()

        # ---- 3. 导纳滤波 ----
        q_adm = self.admittance.update(q_policy, tau_ext)

        # ---- 4. 指令限速 ----
        delta = q_adm - self.q_last
        max_delta = self.cfg.safety.max_command_delta
        over_limit = np.abs(delta) > max_delta
        delta[over_limit] = np.clip(delta[over_limit], -max_delta, max_delta)
        q_cmd = self.q_last + delta
        self.delta_cmd = delta.copy()

        # ---- 5. 安全位置限位 ----
        q_cmd = np.clip(q_cmd, self.cfg.safety.pos_min, self.cfg.safety.pos_max)

        # ---- 6. 紧急停止检测 ----
        tau_mag = np.linalg.norm(tau_ext)
        if tau_mag > self.cfg.safety.emergency_stop_tau:
            print(f"[SAFETY] EMERGENCY STOP! tau_norm={tau_mag:.1f} Nm")
            q_cmd = self.q_last.copy()  # 停止

        self.q_last = q_cmd.copy()
        self.step_count += 1
        return q_cmd

    def set_task_phase(self, phase: int):
        """切换任务相位 (变量导纳)."""
        self.cfg.task_phase = phase
        if isinstance(self.admittance, VariableAdmittance):
            self.admittance.set_phase(phase)
        print(f"[Controller] Phase → {phase}")

    @property
    def tau_external(self) -> np.ndarray:
        return self.tau_ext_last.copy()

    @property
    def est_force(self) -> float:
        """估计末端力大小 (N)."""
        return float(np.linalg.norm(self.tau_ext_last))

    @property
    def last_gripper(self) -> float:
        """最后一次 step() 中的夹爪指令 (0-255)."""
        return self._last_gripper
