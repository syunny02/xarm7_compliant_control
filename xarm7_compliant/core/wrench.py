#!/usr/bin/env python3
"""
wrench.py — 六维力/力矩 → 关节力矩映射
=========================================
核心: τ = J(q)^T · [Fx, Fy, Fz, Tx, Ty, Tz]^T

三种模式:
  1. MuJoCo 模式: 使用 mj_jacBody 计算精确几何 Jacobian
  2. 真机 ROS2 模式: 通过 KDL/TF 获取 Jacobian
  3. 近似模式: 经验公式 (快速原型)

真机部署:
  - xArm7 通常在腕部安装 FT 传感器 (如 OnRobot HEX, Robotiq FT300)
  - 需要机器人 URDF/SRDF 中的运动学模型计算 Jacobian
  - ROS2 中通过 robot_state_publisher + tf2_kdl 获取
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


@dataclass
class WrenchConfig:
    """力/力矩传感器配置."""
    sensor_frame: str = "link7"          # 传感器安装坐标系
    force_scale: float = 1.0             # 力缩放系数 (标定用)
    torque_scale: float = 1.0            # 力矩缩放系数
    bias: np.ndarray = None              # 零点偏置 (shape 6,)
    filter_alpha: float = 0.3            # 低通滤波系数 (0-1, 越大越平滑)
    max_force_n: float = 200.0           # 最大安全力 (N)
    max_torque_nm: float = 30.0          # 最大安全力矩 (Nm)
    gravity_comp: bool = True            # 是否启用重力补偿

    # 重力补偿参数 (真机模式用)
    ee_mass: float = 0.5                 # 末端质量 (kg), 含夹爪
    ee_com: tuple = (0.0, 0.0, -0.05)    # CoM 相对传感器 (m)
    gravity: tuple = (0.0, 0.0, -9.81)   # 重力加速度向量

    def __post_init__(self):
        if self.bias is None:
            self.bias = np.zeros(6)

    @property
    def max_wrench(self) -> np.ndarray:
        return np.array([self.max_force_n] * 3 + [self.max_torque_nm] * 3)


class WrenchMapper:
    tau_external: np.ndarray = None  # 最后一次计算的关节力矩
    """六维力/力矩到关节力矩的映射.

    Args:
        model: MuJoCo MjModel (仿真模式) 或 None (真机模式)
        data: MuJoCo MjData (仿真模式) 或 None (真机模式)
        body_name: 计算 Jacobian 的参考坐标系 (link7/腕部)
        cfg: 传感器配置
        jacobian_fn: 真机模式下外部传入的 Jacobian 计算函数
    """

    def __init__(
        self,
        model=None,
        data=None,
        body_name: str = "link7",
        cfg: Optional[WrenchConfig] = None,
        jacobian_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.data = data
        self.body_name = body_name
        self.cfg = cfg or WrenchConfig()
        self.jacobian_fn = jacobian_fn
        self._body_id = None
        self._site_id = -1
        self._jac = None
        self._wrench_filtered = np.zeros(6)
        self._filter_initialized = False

        if model is not None:
            import mujoco
            # 先查 body，再查 site
            self._body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if self._body_id < 0:
                self._site_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_SITE, body_name
                )
                if self._site_id >= 0:
                    # 用 site 的 parent body
                    self._body_id = model.site_bodyid[self._site_id]
            if self._body_id < 0:
                print(f"[wrench] WARNING: body '{body_name}' not found, using body 0")
                self._body_id = 0
            self._jac = np.zeros((6, model.nv))

    def bias_wrench(self, raw: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """采集零点偏置 (真机标定用).
        
        在机器人静止且无接触时调用, 采集 n_samples 帧取平均.
        """
        samples = []
        for _ in range(n_samples):
            samples.append(raw.copy())
        bias = np.median(samples, axis=0)
        self.cfg.bias = bias
        return bias

    def compensate_gravity(
        self, q: np.ndarray, raw: np.ndarray
    ) -> np.ndarray:
        """重力补偿: 减去末端质量产生的重力分量.

        原理:
          F/T 传感器读数 = 接触力 + 末端重力分量
          重力分量 = R_sensor(q)ᵀ · (m_ee · g)   (力部分)
                    + r_com × (R_sensor(q)ᵀ · m_ee · g)  (力矩部分)

        MuJoCo 模式: 使用 mj_rne / qfrc_bias 精确计算
        真机模式:   用正运动学 + 末端质量/CoM 近似

        Args:
            q:  当前关节位置 (ndof,)
            raw: 去偏置后的原始 wrench (6,)

        Returns:
            gravity_compensated: 减去重力后的净 wrench (6,)
        """
        if not self.cfg.gravity_comp:
            return raw

        # MuJoCo 模式: 使用 qfrc_bias (逆动力学精确重力矩)
        if self.model is not None and self.data is not None:
            import mujoco
            ndof = len(q)
            self.data.qpos[:ndof] = q[:ndof]
            mujoco.mj_forward(self.model, self.data)
            # qfrc_bias 是重力 + 科氏力的 joint torque
            tau_gravity = self.data.qfrc_bias[:ndof].copy()
            # 只取重力部分: 在零速下 qfrc_bias ≈ 重力矩
            # 转换为 wrench: F_gravity ≈ J_pinvᵀ · τ_gravity
            # 但 J 可能非方阵，用伪逆
            jac = np.zeros((6, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data,
                              jac[:3], jac[3:], self._body_id)
            J = jac[:, :ndof]
            J_pinvT = np.linalg.pinv(J).T  # (ndof, 6)ᵀ → (6, ndof)
            wrench_gravity = J_pinvT @ tau_gravity  # (6,)
            return raw - wrench_gravity

        # 真机模式: 使用正运动学近似
        try:
            g = np.array(self.cfg.gravity, dtype=np.float64)  # (3,)
            mass = self.cfg.ee_mass
            com = np.array(self.cfg.ee_com, dtype=np.float64)  # (3,)

            # 计算传感器坐标系旋转 (从基座到传感器)
            # 这里假设已通过 FK 得到旋转; 如果没有 FK, 用近似
            R = self._estimate_sensor_rotation(q)

            # 重力在传感器坐标系的分量
            g_sensor = R.T @ g  # (3,)
            f_gravity = mass * g_sensor  # (3,)
            tau_gravity = np.cross(com, f_gravity)  # (3,)

            wrench_gravity = np.zeros(6)
            wrench_gravity[:3] = f_gravity
            wrench_gravity[3:] = tau_gravity
            return raw - wrench_gravity

        except Exception as e:
            # 保证重力补偿失败不崩溃
            return raw

    def _estimate_sensor_rotation(self, q: np.ndarray) -> np.ndarray:
        """用正运动学估计传感器坐标系的旋转矩阵 (3×3).

        MuJoCo 模式: 直接从 data.site_xmat 获取
        真机模式: 使用近似旋转 (忽略肩部偏移，仅考虑末端朝向)

        Returns:
            R: 3×3 旋转矩阵, 从传感器坐标系到世界坐标系
        """
        if self.data is not None:
            # MuJoCo: 从 site/body 位置获取旋转
            import mujoco
            ndof = len(q)
            self.data.qpos[:ndof] = q[:ndof]
            mujoco.mj_forward(self.model, self.data)
            if self._body_id >= 0:
                mat = self.data.xmat[self._body_id].reshape(3, 3).copy()
                return mat
            return np.eye(3)

        # 真机模式: 近似旋转 (假设只有 joint3/4/5 影响末端朝向)
        if len(q) >= 5:
            c3, s3 = np.cos(q[2]), np.sin(q[2])
            c4, s4 = np.cos(q[3]), np.sin(q[3])
            c5, s5 = np.cos(q[4]), np.sin(q[4])
            Rx = np.array([[1, 0, 0], [0, c3, -s3], [0, s3, c3]])
            Ry = np.array([[c4, 0, s4], [0, 1, 0], [-s4, 0, c4]])
            Rz = np.array([[c5, -s5, 0], [s5, c5, 0], [0, 0, 1]])
            return Rz @ Ry @ Rx
        return np.eye(3)

    def filter_wrench(self, raw: np.ndarray) -> np.ndarray:
        """一阶低通滤波."""
        alpha = self.cfg.filter_alpha
        if not self._filter_initialized:
            self._wrench_filtered = raw.copy()
            self._filter_initialized = True
        else:
            self._wrench_filtered = (
                alpha * raw + (1 - alpha) * self._wrench_filtered
            )
        return self._wrench_filtered.copy()

    def apply_safety_clamp(self, wrench: np.ndarray) -> np.ndarray:
        """安全限幅."""
        max_w = self.cfg.max_wrench
        return np.clip(wrench, -max_w, max_w)

    def compute_tau(
        self, q: np.ndarray, wrench_raw: np.ndarray
    ) -> np.ndarray:
        """计算关节力矩.

        Args:
            q: 当前关节位置 (ndof,)
            wrench_raw: 原始六维力/力矩 (6,)

        Returns:
            tau: 关节力矩 (ndof,)
        """
        # 1. 去偏置
        w = wrench_raw - self.cfg.bias

        # 2. 重力补偿
        w = self.compensate_gravity(q, w)

        # 3. 缩放
        w[:3] *= self.cfg.force_scale
        w[3:] *= self.cfg.torque_scale

        # 4. 滤波
        w = self.filter_wrench(w)

        # 5. 安全限幅
        w = self.apply_safety_clamp(w)

        # 6. Jacobian 映射
        tau = self._map_to_torque(q, w)
        self.tau_external = tau
        return tau

    def _map_to_torque(self, q: np.ndarray, wrench: np.ndarray) -> np.ndarray:
        """τ = J(q)^T · wrench."""
        ndof = len(q)

        if self.model is not None and self.data is not None:
            # ---- MuJoCo 精确 Jacobian ----
            import mujoco
            self.data.qpos[:ndof] = q
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_jacBody(
                self.model, self.data,
                self._jac[:3], self._jac[3:], self._body_id
            )
            J = self._jac[:, :ndof]  # (6, ndof)
            tau = J.T @ wrench
            return tau

        elif self.jacobian_fn is not None:
            # ---- 真机外部 Jacobian 函数 ----
            J = self.jacobian_fn(q)
            return J.T @ wrench

        else:
            # ---- 近似映射 (仅原型验证) ----
            # 粗略经验公式, 真机需替换为 KDL Jacobian
            f = wrench
            tau_approx = np.array([
                f[2] + 0.5 * f[1],           # joint1
                f[0] + 0.5 * f[5],           # joint2
                f[1] + 0.5 * f[3],           # joint3
                f[2] + 0.5 * f[4],           # joint4
                f[0] + 0.3 * f[5],           # joint5
                f[1] + 0.3 * f[3],           # joint6
                f[5],                         # joint7
            ])
            return tau_approx[:ndof]



