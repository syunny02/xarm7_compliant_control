#!/usr/bin/env python3
"""调试: 检查 MuJoCo Jacobian 是否正确."""
import sys
sys.path.insert(0, ".")

import numpy as np
import mujoco
from xarm7_compliant.controller import (
    CompliantController, CompliantControllerConfig,
    AdmittanceConfig, WrenchConfig, SafetyLimits,
)

cfg = CompliantControllerConfig(
    ndof=7, use_mujoco=True,
    admittance=AdmittanceConfig(max_delta_rad=0.15),
)
ctrl = CompliantController(cfg)
ctrl.reset(np.zeros(7))

# 检查 MuJoCo 模型信息
if ctrl._mujoco is not None:
    m = ctrl._mujoco
    d = ctrl._mujoco_data
    print(f"model.nq={m.nq}, model.nv={m.nv}")
    print(f"model.nbody={m.nbody}")
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  body[{i}]: {name}")
    
    print(f"\ndata.qpos shape={d.qpos.shape}")
    print(f"data.qpos={d.qpos}")

# 检查 wrench mapper 中的 jac
mapper = ctrl.wrench
print(f"\nmapper._jac shape={mapper._jac.shape}")
print(f"mapper._body_id={mapper._body_id}")
if mapper._body_id != -1:
    body_name = mujoco.mj_id2name(mapper.model, mujoco.mjtObj.mjOBJ_BODY, mapper._body_id)
    print(f"body_name = '{body_name}'")

# 手动计算一次 tau
w = np.array([0, 0, 20.0, 0, 0, 0])
tau = mapper.compute_tau(np.zeros(7), w)
print(f"\ntau_computed = {tau}")
print(f"tau_external = {ctrl.tau_external}")

# 再调 step
q_cmd = ctrl.step(np.zeros(7), w, obs=None)
dq = q_cmd - np.zeros(7)
print(f"\nstep -> dq = {dq}")
print(f"max|dq| = {np.max(np.abs(dq)):.6f}")

# 如果 tau 为零，打印 jac
if np.all(np.abs(tau) < 1e-10):
    print("\n!!! tau 全零! 打印 Jacobian...")
    # 手动计算 Jacobian
    jac = np.zeros((6, ctrl._mujoco.nv))
    ctrl._mujoco_data.qpos[:] = np.zeros(7)
    mujoco.mj_forward(ctrl._mujoco, ctrl._mujoco_data)
    mujoco.mj_jacBody(ctrl._mujoco, ctrl._mujoco_data, jac[:3], jac[3:], mapper._body_id)
    print(f"J (6x{ctrl._mujoco.nv}):\n{np.array2string(jac, precision=4, suppress_small=True)}")
    print(f"J^T @ w:\n{jac.T @ w}")

