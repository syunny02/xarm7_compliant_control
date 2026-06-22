"""
xarm7_compliant — xArm7 柔性控制包
====================================
@author: 强化学习之父 Davied
@project: xarm7_door_ros2

用于 xArm7 开门任务的 Sim-to-Real 柔性控制器。
支持仿真验证与真机部署（ROS2）。

关键组件:
  core.admittance    — 关节空间导纳滤波器 (M·Δq̈ + D·Δq̇ + K·Δq = τ_ext)
  core.policy        — 策略加载与推理 (与 ROS2 policy_loader.py 兼容)
  core.wrench        — 六维力/力矩 → 关节力矩映射 (Jacobian 转置)
  core.jacobian      — Jacobian 提供器 (MuJoCo/KDL/近似)
  controller         — 柔性控制器主流水线 (支持 Cartesian VIC 模式)
  config_loader      — YAML 配置加载器
  ros.node           — ROS2 真机部署节点
"""

__version__ = "1.0.0"

from .controller import CompliantController, CompliantControllerConfig, SafetyLimits
from .config_loader import load_config, load_config_from_string

__all__ = [
    "CompliantController", "CompliantControllerConfig", "SafetyLimits",
    "load_config", "load_config_from_string",
]
