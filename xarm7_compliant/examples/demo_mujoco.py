#!/usr/bin/env python3
"""
demo_mujoco.py — MuJoCo 仿真验证 demo
=======================================
在仿真环境中验证柔性控制器的行为:
  1. 加载 door_env 场景 (或内置测试场景)
  2. 施加外部力扰动, 观察导纳滤波响应
  3. 用 matplotlib 实时绘制

运行:
    cd xarm7_compliant_control
    python -m xarm7_compliant.examples.demo_mujoco

输出:
    - 控制台日志: 各关节位置偏移
    - matplotlib: 实时力-位移曲线
"""
from __future__ import annotations
import time
import numpy as np
from pathlib import Path
import os

from ..controller import (
    CompliantController, CompliantControllerConfig,
    AdmittanceConfig, WrenchConfig, SafetyLimits,
)


def find_scene_xml() -> str:
    """尝试查找 door_scene.xml."""
    candidates = [
        "../mujoco_rl/door_scene.xml",
        "../../mujoco_rl/door_scene.xml",
        "mujoco_rl/door_scene.xml",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            print(f"[demo] Using scene: {p.resolve()}")
            return str(p.resolve())
    return ""


def generate_test_wrench(t: float) -> np.ndarray:
    """生成测试用模拟外力.

    0-5s: 无外力
    5-10s: 恒定推拉力 (正向 Z)
    10-15s: 交变力
    15-20s: 脉冲力
    """
    w = np.zeros(6)
    if t < 5.0:
        pass  # 空闲
    elif t < 10.0:
        w[2] = 20.0   # Fz = 20N (推)
    elif t < 15.0:
        w[2] = 15.0 * np.sin(t * 0.5)   # 交变力
        w[0] = 5.0 * np.sin(t * 0.8)    # 横向扰动
    elif t < 20.0:
        if t % 2 < 0.3:  # 脉冲
            w[2] = 50.0
    return w


def run_demo():
    """运行 MuJoCo 仿真柔性控制 demo."""
    import matplotlib.pyplot as plt

    # 配置
    cfg = CompliantControllerConfig(
        ndof=7,
        policy_path="",             # 无策略: 保持在初始位置
        device="cpu",
        control_dt=0.01,            # 100Hz
        use_mujoco=True,            # 仿真模式
        scene_xml=find_scene_xml(),
        admittance=AdmittanceConfig(
            mass=1.0,
            damping=12.0,
            stiffness=40.0,
            max_delta_rad=0.15,
            dt=0.01,
        ),
        wrench=WrenchConfig(filter_alpha=0.3),
        safety=SafetyLimits(),
    )

    ctrl = CompliantController(cfg)

    # 初始位置 (零位)
    q_home = np.zeros(7)
    q_home[2] = 1.5   # 关节3 略微抬起 (模拟开门姿势)
    ctrl.reset(q_home)

    # 日志
    log_t, log_fz, log_dq7, log_tau7 = [], [], [], []

    print("=" * 60)
    print("xArm7 柔性控制 MuJoCo 仿真验证")
    print("=" * 60)
    print(f"  导纳参数: M={cfg.admittance.mass}, D={cfg.admittance.damping}, "
          f"K={cfg.admittance.stiffness}")
    print(f"  外力: 0-5s 空闲 | 5-10s Fz=20N | 10-15s 交变 | 15-20s 脉冲")
    print("-" * 60)

    T = 20.0  # 仿真时长 (秒)
    steps = int(T / cfg.control_dt)

    for s in range(steps):
        t = s * cfg.control_dt

        # 生成测试外力
        wrench = generate_test_wrench(t)

        # 控制步 (保持原位, 验证导纳响应)
        q_cmd = ctrl.step(q_home, wrench, obs=None)

        # 偏移量
        dq = q_cmd - q_home

        # 日志
        log_t.append(t)
        log_fz.append(wrench[2])
        log_dq7.append(dq[6].copy())   # 关节7 偏移
        log_tau7.append(ctrl.tau_external[6].copy())

        if s % 200 == 0:  # 每 2s 输出一次
            tau_mag = np.linalg.norm(ctrl.tau_external)
            dq_max = np.max(np.abs(dq))
            print(f"  t={t:5.1f}s | Fz={wrench[2]:6.1f}N | "
                  f"Δq_max={dq_max:.4f}rad | ||τ||={tau_mag:.2f}Nm")

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(log_t, log_fz, label="Fz (外力)", color="red", linewidth=1.5)
    ax1.set_ylabel("Force (N)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(log_t, log_dq7, label="Δq₇ (关节7偏移)", color="blue", linewidth=1.5)
    ax2.set_ylabel("Δq (rad)")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(log_t, log_tau7, label="τ₇_ext (外部关节力矩)", color="green", linewidth=1.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque (Nm)")
    ax3.legend()
    ax3.grid(True)

    fig.suptitle("xArm7 导纳控制仿真验证 (M·Δq̈ + D·Δq̇ + K·Δq = τ_ext)")
    plt.tight_layout()
    plt.savefig("compliant_demo_mujoco.png", dpi=150)
    print(f"\n[demo] 曲线已保存: compliant_demo_mujoco.png")
    plt.show()


def run_demo_vic():
    """Cartesian VIC 模式仿真 demo.

    模拟 RL 策略输出 [dx, K, gripper]，验证 IK + 导纳全链路.
    """
    import matplotlib.pyplot as plt

    from ..config_loader import load_config

    # 从 YAML 加载配置
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "door_opening.yaml"
    )
    if os.path.exists(config_path):
        cfg = load_config(config_path)
        print(f"[demo] Loaded config: {config_path}")
    else:
        print(f"[demo] Config not found, using defaults")
        cfg = CompliantControllerConfig(
            ndof=7,
            control_dt=0.01,
            use_mujoco=True,
            cartesian_vic=True,
            stiffness_from_policy=True,
            admittance=AdmittanceConfig(
                mass=1.0, damping=12.0, stiffness=40.0,
                max_delta_rad=0.15, dt=0.01, stiffness_from_policy=True,
            ),
        )

    ctrl = CompliantController(cfg)
    q_home = np.zeros(7)
    q_home[2] = 0.5  # 初始位
    ctrl.reset(q_home)

    log_t, log_dx, log_dq_max, log_K, log_tau = [], [], [], [], []

    print("=" * 60)
    print("xArm7 Cartesian VIC 仿真验证")
    print("=" * 60)
    print("  策略: 模拟 VIC 输出 [dx, K, gripper]")
    print("  阶段: 0-2s idle | 2-4s 前推 | 4-6s 旋转 | 6-8s 回拉")
    print("-" * 60)

    T = 10.0
    steps = int(T / cfg.control_dt)

    for s in range(steps):
        t = s * cfg.control_dt

        # 模拟策略输出
        if t < 2.0:
            dx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            K = 200.0
        elif t < 4.0:
            dx = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])  # 前推 2cm/s
            K = 100.0  # 接近时低刚度
        elif t < 6.0:
            dx = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0])  # 绕 y 旋转
            K = 200.0
        elif t < 8.0:
            dx = np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 回拉
            K = 300.0
        else:
            dx = np.zeros(6)
            K = 500.0

        # 组装 vic_action
        vic_action = np.array([*dx, K, 0.0])

        # 模拟外力 (开门时的接触力)
        if 3.0 < t < 7.0:
            wrench = np.array([5.0, 0.0, 10.0, 0.0, 0.0, 0.0])  # 接触力
        else:
            wrench = np.zeros(6)

        q_cmd = ctrl.step(q_home, wrench, vic_action=vic_action)

        log_t.append(t)
        log_dx.append(dx[0])
        log_dq_max.append(np.max(np.abs(q_cmd - q_home)))
        log_K.append(ctrl.admittance.cfg.stiffness if hasattr(ctrl.admittance, 'cfg') else K)
        log_tau.append(np.linalg.norm(ctrl.tau_external) if ctrl.tau_external is not None else 0)

        if s % 200 == 0:
            print(f"  t={t:4.1f}s | dx={dx[0]:+.3f} | K={K:.0f} | "
                  f"Δq_max={log_dq_max[-1]:.4f}")

    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(log_t, log_dx, label="dx (策略输出)", color="purple")
    ax1.set_ylabel("dx (m)"); ax1.legend(); ax1.grid(True)

    ax2.plot(log_t, log_dq_max, label="max|Δq| (关节偏移)", color="blue")
    ax2.set_ylabel("Δq (rad)"); ax2.legend(); ax2.grid(True)

    ax3.plot(log_t, log_K, label="K (导纳刚度)", color="orange")
    ax3.set_ylabel("K (N/m)"); ax3.legend(); ax3.grid(True)

    ax4.plot(log_t, log_tau, label="||τ_ext|| (关节力矩)", color="green")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Torque (Nm)")
    ax4.legend(); ax4.grid(True)

    fig.suptitle("xArm7 Cartesian VIC 全链路仿真 (策略→IK→导纳→安全)")
    plt.tight_layout()
    plt.savefig("compliant_demo_vic.png", dpi=150)
    print(f"\n[demo] VIC模式曲线: compliant_demo_vic.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="xArm7 Compliant Control Demo")
    parser.add_argument("--vic", action="store_true",
                        help="Run Cartesian VIC mode demo")
    args = parser.parse_args()
    if args.vic:
        run_demo_vic()
    else:
        run_demo()
