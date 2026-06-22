"""
jacobian_provider.py — Jacobian 矩阵提供器
============================================
为 Cartesian VIC 策略的差分 IK 提供三种 Jacobian 实现：

  1. MuJoCoJacobianProvider  — 仿真模式，使用 MuJoCo 的 mj_jacSite 精确计算
  2. KDLJacobianProvider     — 真机模式，使用 KDL 运动学链 (待 install_requires 加 pykdl)
  3. ApproxJacobianProvider  — 快速原型，经验近似公式

用法:
    provider = MuJoCoJacobianProvider(model, data, "link_tcp", ndof=7)
    J = provider.get_jacobian(q)   # 返回 (6, ndof)
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class JacobianProviderBase:
    """Jacobian 提供器基类. 子类必须实现 get_jacobian(q)."""
    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MuJoCoJacobianProvider(JacobianProviderBase):
    """MuJoCo 精确 Jacobian.

    使用 MuJoCo 的 mj_jacSite 计算末端执行器的 (6×nv) Jacobian.
    适用于仿真模式，可直接用于差分 IK.
    """
    def __init__(self, model, data, site_name: str = "link_tcp", ndof: int = 7):
        import mujoco
        self._model = model
        self._data = data
        self._ndof = ndof
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            # fallback: 使用 body 坐标
            self._use_site = False
            self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_name)
            if self._body_id < 0:
                raise ValueError(f"Site/body '{site_name}' not found in MuJoCo model")
        else:
            self._use_site = True
        self._jac = np.zeros((6, model.nv))

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        import mujoco
        ndof = self._ndof
        self._data.qpos[:ndof] = q[:ndof]
        mujoco.mj_forward(self._model, self._data)
        if self._use_site:
            mujoco.mj_jacSite(
                self._model, self._data,
                self._jac[:3], self._jac[3:], self._site_id,
            )
        else:
            mujoco.mj_jacBody(
                self._model, self._data,
                self._jac[:3], self._jac[3:], self._body_id,
            )
        return self._jac[:, :ndof].copy()


class KDLJacobianProvider(JacobianProviderBase):
    """ROS2 KDL 真机 Jacobian.

    从 robot_description 参数构建 KDL 树，
    通过 ChainJntToJac 计算 6×ndof Jacobian.

    需要: urdfdom_py, PyKDL, kdl_parser_py
    安装: pip install urdfdom_py pykdl_utils
    """
    def __init__(self, robot_description: str, base_link: str = "link0",
                 tip_link: str = "link_tcp", ndof: int = 7):
        try:
            from urdf_parser_py.urdf import URDF
            from kdl_parser_py import kdl_tree_from_urdf_model
            import PyKDL
        except ImportError:
            raise ImportError(
                "KDL Jacobian 需要: pip install urdfdom_py pykdl_utils\n"
                "或使用 ApproxJacobianProvider 作为 fallback"
            )
        self._ndof = ndof
        self._robot_desc = robot_description

        # 解析 URDF → KDL tree
        urdf = URDF.from_xml_string(robot_description)
        tree = kdl_tree_from_urdf_model(urdf)
        chain = tree.getChain(base_link, tip_link)
        self._chain = chain
        self._jac_solver = PyKDL.ChainJntToJacSolver(chain)
        self._q_kdl = PyKDL.JntArray(chain.getNrOfJoints())

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        import PyKDL
        ndof = min(len(q), self._chain.getNrOfJoints())
        for i in range(ndof):
            self._q_kdl[i] = float(q[i])
        jac_kdl = PyKDL.Jacobian(self._chain.getNrOfJoints())
        self._jac_solver.JntToJac(self._q_kdl, jac_kdl)
        # PyKDL Jacobian 是 6×n
        J = np.zeros((6, self._ndof))
        for i in range(6):
            for j in range(ndof):
                J[i, j] = jac_kdl[i, j]
        return J


class ApproxJacobianProvider(JacobianProviderBase):
    """近似经验 Jacobian (快速原型用).

    使用 xArm7 的简化运动学近似.
    仅用于无 MuJoCo 也无 KDL 时的快速调试 —— 精度不保证.
    """
    def __init__(self, ndof: int = 7):
        self._ndof = ndof

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """经验近似 Jacobian.

        假设 xArm7 的简化几何:
        - 关节1/2/3 影响 x/y/z 大范围
        - 关节4/5/6 影响 orientation
        - 关节7 是末端旋转 (flange)
        """
        ndof = min(len(q), self._ndof)
        J = np.zeros((6, ndof))
        q_safe = np.nan_to_num(q[:ndof], nan=0.0)

        # 位置部分 (3×ndof) — 基于臂长+关节角的近似
        L = 0.3  # 近似臂长
        J[0, 0] = -L * np.sin(q_safe[0])
        J[0, 1] = L * np.cos(q_safe[1])
        J[0, 2] = L * np.cos(q_safe[2])
        J[1, 0] = L * np.cos(q_safe[0])
        J[1, 1] = L * np.sin(q_safe[1])
        J[1, 2] = L * np.sin(q_safe[2])
        J[2, 1] = 0.3
        J[2, 2] = 0.2

        # 姿态部分 (3×ndof) — J[3:6] 为旋转行
        for i in range(3, min(6, ndof)):
            J[3 + (i - 3), i] = 1.0

        # 安全: 避免奇异矩阵
        rank = np.linalg.matrix_rank(J)
        if rank < min(6, ndof):
            J += np.eye(6, ndof) * 0.01
        return J
