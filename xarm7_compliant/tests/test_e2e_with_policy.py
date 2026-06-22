#!/usr/bin/env python3
"""
test_e2e_with_policy.py — Cartesian VIC 策略全链路端到端测试
==============================================================

测试完整流水线:
  VIC policy → [dx, K, gripper]
      → IK (J_pinv @ dx_cart)
      → joint target (q + dq)
      → admittance filter (Δq from F/T)
      → safety clamp
      → joint command

场景:
  1. 零外力: 策略输出 → 关节指令无漂移
  2. 恒外力: 导纳偏移合理
  3. 策略刚度覆盖: K_policy 正确传递到 admittance
  4. 夹爪指令: gripper 值正确透传

运行:
    python -m xarm7_compliant.tests.test_e2e_with_policy
"""
import sys
import os
import numpy as np
from pathlib import Path

# 添加包路径
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", ".."))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from xarm7_compliant.controller import (
    CompliantController, CompliantControllerConfig,
    AdmittanceConfig, WrenchConfig, SafetyLimits,
)


def find_policy() -> str:
    """查找训练产出的 .pt 策略文件."""
    candidates = [
        os.path.join(HERE, "..", "..", "..", "xarm7_cartesian_vic",
                     "runs", "VIC_PPO_seed0", "policy.pt"),
        os.path.join(HERE, "..", "..", "..", "xarm7_cartesian_vic",
                     "runs", "V3_anneal_300k", "policy.pt"),
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            print(f"[E2E] Using policy: {p}")
            return str(p.resolve())
    return ""


def make_ctrl(policy_path: str = "") -> CompliantController:
    """构造 Cartesian VIC 模式的控制器 (无 MuJoCo，使用近似 Jacobian)."""
    cfg = CompliantControllerConfig(
        ndof=7,
        policy_path=policy_path,
        device="cpu",
        control_dt=0.01,
        use_mujoco=False,           # 无 MuJoCo 依赖
        tcp_body_name="link_tcp",

        # ── Cartesian VIC 模式 ──
        cartesian_vic=True,
        stiffness_from_policy=True,

        admittance=AdmittanceConfig(
            mass=1.0,
            damping=12.0,
            stiffness=40.0,        # 会被策略 K 覆盖
            max_delta_rad=0.15,
            dt=0.01,
            stiffness_from_policy=True,
        ),
        wrench=WrenchConfig(filter_alpha=0.3, gravity_comp=False),
        safety=SafetyLimits(max_command_delta=0.05),
    )
    return CompliantController(cfg)


def test_zero_force_drift():
    """场景1: 零外力时策略动作不应导致漂移."""
    print("\n" + "=" * 60)
    print("[E2E] 测试1: 零力无漂移")
    print("=" * 60)

    ctrl = make_ctrl()
    q_home = np.zeros(7)
    q_home[2] = 0.5  # 非零初始位
    ctrl.reset(q_home)

    # 零外力 + 零动作
    for s in range(100):  # 1 秒
        vic_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0])
        q_cmd = ctrl.step(q_home, np.zeros(6), vic_action=vic_action)
        dq = q_cmd - q_home
        max_dq = np.max(np.abs(dq))
        if s == 50:
            print(f"  t=0.5s: max|Δq|={max_dq:.6f} rad")

    max_dq = np.max(np.abs(ctrl.q_last - q_home))
    print(f"  最终 max|Δq|={max_dq:.6f} rad")

    if max_dq < 0.02:
        print(f"  ✅ 通过: 零力漂移 {max_dq:.6f} < 0.02 rad")
    else:
        print(f"  ❌ 失败: 零力漂移 {max_dq:.6f} >= 0.02 rad")
    return max_dq < 0.02


def test_constant_force_response():
    """场景2: 恒外力产生恒定导纳偏移."""
    print("\n" + "=" * 60)
    print("[E2E] 测试2: 恒外力导纳响应")
    print("=" * 60)

    # 用高刚度排除策略变动影响
    ctrl = make_ctrl()
    q_home = np.zeros(7)
    ctrl.reset(q_home)

    # 恒外力: Fz=20N (仿真中转为关节力矩)
    wrench_fz = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
    vic_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0])

    for s in range(200):  # 2 秒
        q_cmd = ctrl.step(q_home, wrench_fz, vic_action=vic_zero)

    dq = ctrl.q_last - q_home
    max_dq = np.max(np.abs(dq))
    print(f"  恒力 Fz=20N → 最大偏移 {max_dq:.4f} rad")

    # 偏移应 > 0 (有响应) 且 < max_delta (被限幅)
    if 0.001 < max_dq < 0.16:
        print(f"  ✅ 通过: 偏移 {max_dq:.4f} rad 在合理范围")
    else:
        print(f"  ❌ 失败: 偏移 {max_dq:.4f} rad 异常")
    return 0.001 < max_dq < 0.16


def test_stiffness_from_policy():
    """场景3: 策略刚度覆盖导纳刚度."""
    print("\n" + "=" * 60)
    print("[E2E] 测试3: 策略刚度覆盖")
    print("=" * 60)

    ctrl = make_ctrl()
    q_home = np.zeros(7)
    ctrl.reset(q_home)

    # 低刚度 K=50
    vic_lowK = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0])
    wrench = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for s in range(200):
        q_cmd = ctrl.step(q_home, wrench, vic_action=vic_lowK)
    dq_low = ctrl.q_last - q_home
    max_low = np.max(np.abs(dq_low))

    # 重置
    ctrl.reset(q_home)

    # 高刚度 K=800
    vic_highK = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0, 0.0])
    for s in range(200):
        q_cmd = ctrl.step(q_home, wrench, vic_action=vic_highK)
    dq_high = ctrl.q_last - q_home
    max_high = np.max(np.abs(dq_high))

    print(f"  低刚度 K=50  → 偏移 {max_low:.4f} rad")
    print(f"  高刚度 K=800 → 偏移 {max_high:.4f} rad")

    # 低K → 大偏移, 高K → 小偏移 (至少趋势正确)
    if max_low > max_high:
        print(f"  ✅ 通过: 低K偏移({max_low:.4f}) > 高K偏移({max_high:.4f})")
    else:
        print(f"  ⚠️  注意: 趋势反常 (低K={max_low:.4f} ≤ 高K={max_high:.4f})")
    return max_low > max_high


def test_gripper_pass_through():
    """场景4: 夹爪指令透传."""
    print("\n" + "=" * 60)
    print("[E2E] 测试4: 夹爪指令")
    print("=" * 60)

    ctrl = make_ctrl()
    q_home = np.zeros(7)
    ctrl.reset(q_home)

    vic_action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 500.0, 128.0])
    ctrl.step(q_home, np.zeros(6), vic_action=vic_action)

    gripper = ctrl.last_gripper
    print(f"  夹爪指令 = {gripper}")

    if abs(gripper - 128.0) < 0.01:
        print(f"  ✅ 通过: gripper={gripper} 正确透传")
    else:
        print(f"  ❌ 失败: gripper={gripper}, 期望 128.0")
    return abs(gripper - 128.0) < 0.01


def test_cartesian_delta_ik():
    """场景5: 笛卡尔增量 → IK → 关节动作."""
    print("\n" + "=" * 60)
    print("[E2E] 测试5: 笛卡尔 IK")
    print("=" * 60)

    ctrl = make_ctrl()
    q_home = np.zeros(7)
    q_home[2] = 0.5  # 关节3抬起
    ctrl.reset(q_home)

    # 沿 x 正向 1cm
    vic_action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0])
    q_cmd = ctrl.step(q_home, np.zeros(6), vic_action=vic_action)

    dq = q_cmd - q_home
    dq_norm = np.linalg.norm(dq)
    print(f"  vic_action dx=0.01m → Δq_norm={dq_norm:.4f} rad")

    if dq_norm > 0.001 and not np.any(np.isnan(dq)):
        print(f"  ✅ 通过: IK 产生有效关节位移")
    else:
        print(f"  ❌ 失败: IK 无输出或 NaN")
    return dq_norm > 0.001 and not np.any(np.isnan(dq))


def test_safety_clamp():
    """场景6: 大幅度动作被安全限幅."""
    print("\n" + "=" * 60)
    print("[E2E] 测试6: 安全限幅")
    print("=" * 60)

    ctrl = make_ctrl()
    q_home = np.zeros(7)
    ctrl.reset(q_home)

    # 超大幅度动作 (0.5m — 远超限速)
    vic_action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0])
    q_cmd = ctrl.step(q_home, np.zeros(6), vic_action=vic_action)

    dq = q_cmd - q_home
    max_delta = ctrl.cfg.safety.max_command_delta
    max_dq = np.max(np.abs(dq))

    print(f"  单步最大变化 = {max_dq:.4f} rad (限幅 {max_delta})")

    if np.all(np.abs(dq) <= max_delta * 1.01):
        print(f"  ✅ 通过: 指令被限幅到 {max_delta} rad/step")
    else:
        print(f"  ❌ 失败: 指令超限")
    return np.all(np.abs(dq) <= max_delta * 1.01)


if __name__ == "__main__":
    results = {}

    try:
        results["zero_drift"] = test_zero_force_drift()
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        results["zero_drift"] = False

    try:
        results["force_response"] = test_constant_force_response()
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        results["force_response"] = False

    try:
        results["stiffness"] = test_stiffness_from_policy()
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        results["stiffness"] = False

    try:
        results["gripper"] = test_gripper_pass_through()
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        results["gripper"] = False

    try:
        results["ik"] = test_cartesian_delta_ik()
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        results["ik"] = False

    try:
        results["safety"] = test_safety_clamp()
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        results["safety"] = False

    # 汇总
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  [{passed}/{total}] 测试通过")
    for name, ok in results.items():
        print(f"    {name}: {'✅' if ok else '❌'}")
    print("=" * 60)
