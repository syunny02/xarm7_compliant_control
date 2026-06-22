#!/usr/bin/env python3
"""
demo_real_robot.py — 真机部署测试脚本
=======================================
用于直接在真机 xArm7 上测试柔性控制.

运行前:
  1. 确保 xArm7 已上电, 处于 E-stop 释放状态
  2. 确保 F/T 传感器已连接并发布 /force_torque_sensor/wrench
  3. 确保 joint_states 正常发布

运行:
    python -m xarm7_compliant.examples.demo_real_robot --policy model.pt

安全提示:
  - 首次运行务必在"仿真模式"验证参数 (use_sim=True)
  - 真机运行时, 手放在 E-stop 按钮上
  - 从低导纳增益开始 (高阻尼, 低刚度)
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from pathlib import Path

from ..controller import (
    CompliantController, CompliantControllerConfig,
    AdmittanceConfig, WrenchConfig, SafetyLimits,
)
from ..sim_to_real import SimToRealConfig, SimToRealWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="xArm7 真机柔性控制测试")
    parser.add_argument("--policy", type=str, default="",
                        help="策略文件路径 (.pt)")
    parser.add_argument("--use-sim", action="store_true",
                        help="仿真模式 (不连接真机)")
    parser.add_argument("--damping", type=float, default=15.0,
                        help="导纳阻尼 (越大越硬, 推荐 12-20)")
    parser.add_argument("--stiffness", type=float, default=30.0,
                        help="导纳刚度 (越大越硬, 推荐 20-60)")
    parser.add_argument("--mass", type=float, default=1.0,
                        help="导纳质量 (推荐 0.5-2.0)")
    parser.add_argument("--max-delta", type=float, default=0.10,
                        help="最大位置偏移 (rad)")
    parser.add_argument("--freq", type=float, default=100.0,
                        help="控制频率 (Hz)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 安全检查
    if not args.use_sim:
        print("\n" + "!" * 50)
        print("  真机模式! 请确保:")
        print("    1. xArm7 已就绪, E-stop 已释放")
        print("    2. 手放在 E-stop 按钮上")
        print("    3. 从低导纳增益开始")
        print("!" * 50 + "\n")
        resp = input("确认继续? (yes/no): ")
        if resp.lower() not in ("yes", "y"):
            print("退出.")
            return

    # 配置
    cfg = CompliantControllerConfig(
        ndof=7,
        policy_path=args.policy,
        device="cpu",
        control_dt=1.0 / args.freq,
        use_mujoco=args.use_sim,
        admittance=AdmittanceConfig(
            mass=args.mass,
            damping=args.damping,
            stiffness=args.stiffness,
            max_delta_rad=args.max_delta,
            dt=1.0 / args.freq,
        ),
    )

    ctrl = CompliantController(cfg)

    print("=" * 60)
    print("xArm7 真机柔性控制测试")
    print("=" * 60)
    print(f"  阻尼(D)={args.damping}, 刚度(K)={args.stiffness}, "
          f"质量(M)={args.mass}")
    print(f"  最大偏移={args.max_delta} rad")
    print(f"  策略: {args.policy or '无 (位置保持)'}")
    print(f"  模式: {'仿真' if args.use_sim else '真机'}")
    print("-" * 60)
    print("可用命令: q=退出 | r=重置 | p+/-=切换相位 | "
          "d+/−=调阻尼 | k+/−=调刚度")
    print("=" * 60)

    # 仿真模式: 模拟 F/T 输入
    t = 0.0
    q_home = np.zeros(7)

    try:
        while True:
            # 仿真模式下模拟外力
            if args.use_sim:
                wrench = np.zeros(6)
                # 20N 恒推力 + 5N 横向扰动
                wrench[2] = 20.0 + 5.0 * np.sin(t * 0.5)
                q_cmd = ctrl.step(q_home, wrench)
            else:
                # 真机模式下通过 ROS2 节点运行
                print("真机模式请使用: ros2 run xarm7_compliant compliant_node")
                break

            # 每 50 步输出
            if int(t * args.freq) % 50 == 0:
                dq_max = np.max(np.abs(ctrl.admittance.get_offset())) * 1000
                tau_norm = np.linalg.norm(ctrl.tau_external)
                print(f"  t={t:5.1f}s | Δq_max={dq_max:.1f}mrad | "
                      f"||τ||={tau_norm:.2f}Nm")
                # 控制台输入检测 (非阻塞)
                # 实际真机中通过 ROS2 service 调参

            time.sleep(1.0 / args.freq)
            t += 1.0 / args.freq

    except KeyboardInterrupt:
        print("\n[测试结束]")


if __name__ == "__main__":
    main()
