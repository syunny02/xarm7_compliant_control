#!/usr/bin/env python3
"""验证 MuJoCo demo 模块可导入."""
import sys
sys.path.insert(0, ".")

print("[TEST] 导入 demo_mujoco...")
from xarm7_compliant.examples.demo_mujoco import run_demo, generate_test_wrench, find_scene_xml
print("[PASS] demo_mujoco 导入成功")

print("[TEST] generate_test_wrench...")
for t in [0, 7.5, 12.0, 16.0]:
    w = generate_test_wrench(t)
    print(f"  t={t}: Fz={w[2]:.1f}N")
print("[PASS] wrench 生成正常")

print("[TEST] 导入 demo_real_robot...")
from xarm7_compliant.examples.demo_real_robot import parse_args
print("[PASS] demo_real_robot 导入成功")

print("\n=== ALL DEMO IMPORT TESTS PASSED ===")
