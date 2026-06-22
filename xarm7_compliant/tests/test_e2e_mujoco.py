#!/usr/bin/env python3
"""端到端 MuJoCo 仿真验证 (3秒)."""
import sys
sys.path.insert(0, ".")

import numpy as np
from xarm7_compliant.controller import (
    CompliantController, CompliantControllerConfig,
    AdmittanceConfig, WrenchConfig, SafetyLimits,
)

print("[E2E] 初始化控制器...")
cfg = CompliantControllerConfig(
    ndof=7,
    policy_path="",
    device="cpu",
    control_dt=0.01,
    use_mujoco=True,
    admittance=AdmittanceConfig(mass=1.0, damping=12.0, stiffness=40.0, max_delta_rad=0.15, dt=0.01),
)
ctrl = CompliantController(cfg)

q_home = np.zeros(7)
ctrl.reset(q_home)
print("[E2E] 初始化完成")

# 运行 3 秒 = 300 步
print("[E2E] 运行仿真 (3s, 300步)...")
dq_log = []
tau_log = []
for s in range(300):
    t = s * 0.01
    
    # 模拟外力: 0-1s 空闲, 1-2s 20N, 2-3s 脉冲
    wrench = np.zeros(6)
    if 1.0 <= t < 2.0:
        wrench[0] = 20.0  # Fx=20N (沿手臂径向, 产生力矩)
    elif t >= 2.0 and (t % 0.5) < 0.2:
        wrench[0] = 50.0  # Fx=50N 脉冲
    
    q_cmd = ctrl.step(q_home, wrench, obs=None)
    dq = q_cmd - q_home
    dq_log.append(np.max(np.abs(dq)))
    tau_log.append(np.linalg.norm(ctrl.tau_external))

dq_arr = np.array(dq_log)
tau_arr = np.array(tau_log)

# 验证
# 1-2s: 20N恒力, 稳态偏移 ≈ 20/40 = 0.5 rad (但 max_delta=0.15 会限幅)
offset_peak = float(np.max(dq_arr[100:200]))
print(f"[E2E] 1-2s 恒力20N: max偏移={offset_peak:.4f} rad (限幅0.15)")
assert offset_peak <= 0.16, f"偏移应被限幅到0.15, 但为{offset_peak}"

# 零力时偏移应接近0
offset_idle = float(np.max(dq_arr[:50]))
print(f"[E2E] 0-0.5s 零力: max偏移={offset_idle:.6f} rad")
assert offset_idle < 0.01, f"零力时偏移应接近0, 但为{offset_idle}"

# 脉冲响应
offset_pulse = float(np.max(dq_arr[200:]))
print(f"[E2E] 脉冲50N: max偏移={offset_pulse:.4f} rad (限幅0.15)")
assert offset_pulse <= 0.16, f"脉冲偏移超限: {offset_pulse}"

print(f"\n[E2E] === 端到端 MuJoCo 仿真验证通过 ===")
print(f"[E2E] 零力偏移: {offset_idle:.6f} rad ✓")
print(f"[E2E] 恒力限幅: {offset_peak:.4f} rad (≤0.15) ✓")
print(f"[E2E] 脉冲限幅: {offset_pulse:.4f} rad (≤0.15) ✓")
