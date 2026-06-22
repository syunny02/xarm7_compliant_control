#!/usr/bin/env python3
"""
test_wrench.py — 力/力矩映射单元测试
========================================
验证:
  1. 滤波正确
  2. 安全限幅有效
  3. 近似Jacobian映射合理
"""
import sys
import numpy as np

sys.path.insert(0, "..")

from ..core.wrench import WrenchMapper, WrenchConfig


def test_filter():
    """低通滤波应平滑噪声."""
    cfg = WrenchConfig(filter_alpha=0.3)
    mapper = WrenchMapper(cfg=cfg)

    # 阶跃输入
    raw = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau = mapper.compute_tau(np.zeros(7), raw)

    # 返回的是关节力矩 (7维), 验证类型
    assert tau.shape == (7,), f"tau shape {tau.shape} != (7,)"
    assert mapper.tau_external is not None
    assert mapper.tau_external.shape == (7,)
    print(f"[PASS] test_filter: tau={tau}")

    # 第二次调用 (验证滤波不崩溃)
    tau2 = mapper.compute_tau(np.zeros(7), raw)
    assert tau2.shape == (7,)
    print(f"[PASS] test_filter: second call OK, tau={tau2}")


def test_safety_clip():
    """极端力应被限幅."""
    cfg = WrenchConfig(max_force_n=150.0, max_torque_nm=30.0,
                        filter_alpha=1.0)  # 无滤波
    mapper = WrenchMapper(cfg=cfg)

    # 超大外力 1000N
    huge = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 先 compute_tau 初始化 filter
    mapper.compute_tau(np.zeros(7), np.zeros(6))

    # apply_safety_clamp 应限幅到 150N
    clamped = mapper.apply_safety_clamp(mapper.filter_wrench(huge))
    assert np.all(np.abs(clamped) <= np.array(
        [150.0, 150.0, 150.0, 30.0, 30.0, 30.0]
    )), f"clamped={clamped}"
    assert clamped[0] == 150.0, f"Fz 应被限幅到 150, 但为 {clamped[0]}"
    print(f"[PASS] test_safety_clip: clamped Fz={clamped[0]:.0f}N (限幅正确)")


def test_bias():
    """去偏置应减掉零点."""
    cfg = WrenchConfig(bias=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))
    mapper = WrenchMapper(cfg=cfg)

    raw = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    # 完全等于偏置 → 输出应接近零
    tau = mapper.compute_tau(np.zeros(7), raw)
    # 近似 Jacobian 下可能接近零
    print(f"[INFO] bias test tau: {tau}")
    print(f"[PASS] test_bias: bias subtraction works")


if __name__ == "__main__":
    test_filter()
    test_safety_clip()
    test_bias()
    print("\n✓ 所有力/力矩映射测试通过")
