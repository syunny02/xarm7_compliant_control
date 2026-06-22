#!/usr/bin/env python3
"""
test_admittance.py — 导纳滤波器单元测试
==========================================
验证:
  1. 零外力时输出等于期望位置
  2. 恒外力时输出稳态偏移 = F_ext / K
  3. 变量刚度切换正确
  4. 安全限位有效
"""
import sys
import numpy as np

sys.path.insert(0, "..")

from ..core.admittance import AdmittanceFilter, AdmittanceConfig, VariableAdmittance


def test_zero_force():
    """零外力时导纳输出应等于期望位置."""
    adm = AdmittanceFilter(7)
    q_des = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    adm.reset(q_des)

    tau_zero = np.zeros(7)
    q_cmd = adm.update(q_des, tau_zero)

    assert np.allclose(q_cmd, q_des, atol=1e-6), \
        f"零力时输出应等于期望, 但 Δq={q_cmd - q_des}"
    print("[PASS] test_zero_force")


def test_steady_state_offset():
    """恒外力时稳态偏移 Δq = τ/K."""
    adm = AdmittanceFilter(7, AdmittanceConfig(stiffness=40.0))
    q_des = np.zeros(7)
    adm.reset(q_des)

    # 恒定外部力矩 5 Nm
    tau_const = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 运行 200 步使稳态收敛
    for _ in range(200):
        q_cmd = adm.update(q_des, tau_const)

    dq = q_cmd - q_des
    expected_dq = tau_const[0] / 40.0  # = 5/40 = 0.125 rad

    assert abs(dq[0] - expected_dq) < 0.05, \
        f"偏移 {dq[0]:.4f} 应接近 {expected_dq:.4f}"
    assert abs(dq[1]) < 0.01, \
        f"关节1 (无外力) 偏移应为 0, 但为 {dq[1]:.6f}"
    print(f"[PASS] test_steady_state_offset: dq={dq[0]:.4f} (≈{expected_dq:.4f})")


def test_max_delta_limit():
    """导纳偏移应不超过 max_delta_rad."""
    adm = AdmittanceFilter(7, AdmittanceConfig(max_delta_rad=0.05))
    q_des = np.zeros(7)
    adm.reset(q_des)

    tau_large = np.array([200.0] * 7)

    for _ in range(100):
        q_cmd = adm.update(q_des, tau_large)

    dq = q_cmd - q_des
    assert np.all(np.abs(dq) <= 0.052), \
        f"偏移超限: max |Δq|={np.max(np.abs(dq)):.4f} > 0.05"
    print(f"[PASS] test_max_delta_limit: max|Δq|={np.max(np.abs(dq)):.4f} ≤ 0.05")


def test_variable_admittance():
    """变量导纳相位切换."""
    vadm = VariableAdmittance(7, dt=0.01)
    q_home = np.zeros(7)
    vadm.reset(q_home)

    tau = np.array([1.0] * 7)

    # Phase 0: 高刚度
    vadm.set_phase(0)
    for _ in range(100):
        q0 = vadm.update(q_home, tau)

    # Phase 2: 低刚度
    vadm.set_phase(2)
    for _ in range(100):
        q2 = vadm.update(q_home, tau)

    dq0 = np.max(np.abs(q0 - q_home))
    dq2 = np.max(np.abs(q2 - q_home))
    assert dq2 > dq0, \
        f"Phase 2 (低刚度) 偏移 {dq2:.4f} 应大于 Phase 0 {dq0:.4f}"
    print(f"[PASS] test_variable_admittance: "
          f"Phase0 Δq={dq0:.4f} < Phase2 Δq={dq2:.4f}")


def test_reset():
    """reset 后状态清空."""
    adm = AdmittanceFilter(7)
    q_init = np.array([0.5] * 7)
    adm.reset(q_init)

    assert np.allclose(adm.get_offset(), np.zeros(7), atol=1e-10)
    assert np.allclose(adm.get_velocity(), np.zeros(7), atol=1e-10)
    print("[PASS] test_reset")


if __name__ == "__main__":
    test_zero_force()
    test_steady_state_offset()
    test_max_delta_limit()
    test_variable_admittance()
    test_reset()
    print("\n✓ 所有导纳滤波器测试通过")
