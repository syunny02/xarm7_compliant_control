#!/usr/bin/env python3
"""
node.py — ROS2 真机柔性控制节点
=================================
可直接在真机 xArm7 上部署的 ROS2 节点.

订阅:
  /joint_states                — 关节状态 (sensor_msgs/JointState)
  /force_torque_sensor/wrench  — F/T 传感器 (geometry_msgs/WrenchStamped)
  /policy_action               — 策略动作 (xarm7_door_interfaces/PolicyAction)
  /set_task_mode               — 任务模式 (xarm7_door_interfaces/SetTaskMode)

发布:
  /compliant_joint_cmd         — 柔性关节指令 (std_msgs/Header + 位置/速度)
  /compliant_status            — 状态诊断

启动:
  真机:  ros2 run xarm7_compliant compliant_node --ros-args -p policy_path:=/path/to/model.pt
  仿真:  ros2 run xarm7_compliant compliant_node --ros-args -p use_sim:=true

依赖:
  - rclpy
  - sensor_msgs / geometry_msgs
  - xarm7_door_interfaces (可选)
  - xarm_sdk / xarm_msgs (控制 xArm7 关节)
"""
from __future__ import annotations
import time
import numpy as np
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

from ..controller import CompliantController, CompliantControllerConfig
from ..sim_to_real import SimToRealWrapper, SimToRealConfig
from ..core.wrench import WrenchConfig
from ..core.admittance import AdmittanceConfig


class CompliantControlNode(Node):
    """ROS2 柔性控制节点 — 真机部署入口.

    用法:
        ros2 run xarm7_compliant compliant_node \\
            --ros-args -p policy_path:="/home/user/models/best.pt" \\
                       -p use_sim:=false \\
                       -p control_freq:=100.0
    """

    def __init__(self):
        super().__init__("compliant_control_node")

        # ---- 参数声明 ----
        self.declare_parameters(
            namespace="",
            parameters=[
                ("policy_path", ""),
                ("use_sim", False),
                ("control_freq", 100.0),
                ("admittance_mass", 1.0),
                ("admittance_damping", 12.0),
                ("admittance_stiffness", 40.0),
                ("max_delta_rad", 0.15),
                ("force_threshold", 5.0),
                ("ft_filter_alpha", 0.3),
                ("sim_to_real_friction", 0.15),
            ],
        )

        # ---- 读取参数 ----
        self.policy_path = self.get_parameter("policy_path").value
        self.use_sim = self.get_parameter("use_sim").value
        self.control_freq = self.get_parameter("control_freq").value
        self.dt = 1.0 / self.control_freq

        # ---- 控制器配置 ----
        adm_cfg = AdmittanceConfig(
            mass=self.get_parameter("admittance_mass").value,
            damping=self.get_parameter("admittance_damping").value,
            stiffness=self.get_parameter("admittance_stiffness").value,
            max_delta_rad=self.get_parameter("max_delta_rad").value,
            dt=self.dt,
        )
        wrench_cfg = WrenchConfig(
            filter_alpha=self.get_parameter("ft_filter_alpha").value,
        )
        ctrl_cfg = CompliantControllerConfig(
            ndof=7,
            policy_path=self.policy_path,
            device="cpu",
            control_dt=self.dt,
            use_mujoco=self.use_sim,
            admittance=adm_cfg,
            wrench=wrench_cfg,
        )
        self.controller = CompliantController(ctrl_cfg)

        # ---- Sim-to-Real 包装器 ----
        st_cfg = SimToRealConfig(
            coulomb_friction=self.get_parameter("sim_to_real_friction").value,
        )
        self.sim_to_real = SimToRealWrapper(st_cfg)

        # ---- 状态 ----
        self.q_current = np.zeros(7)
        self.dq_current = np.zeros(7)
        self.wrench_raw = np.zeros(6)
        self.obs_current: Optional[np.ndarray] = None
        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7",
        ]
        self._initialized = False

        # ---- 订阅 ----
        self._sub_joint = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )
        self._sub_wrench = self.create_subscription(
            WrenchStamped, "/force_torque_sensor/wrench", self._wrench_cb, 10
        )

        # ---- 发布 ----
        self._pub_joint_cmd = self.create_publisher(
            JointState, "/compliant_joint_cmd", 10
        )

        # ---- 定时控制循环 ----
        self._timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f"[CompliantControlNode] started | "
            f"freq={self.control_freq}Hz | "
            f"sim={self.use_sim} | "
            f"policy={'loaded' if self.policy_path else 'none'}"
        )

    def _joint_cb(self, msg: JointState):
        """接收 /joint_states 回调."""
        # 按关节名映射
        pos_map = dict(zip(msg.name, msg.position))
        vel_map = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        for i, name in enumerate(self.joint_names):
            if name in pos_map:
                self.q_current[i] = pos_map[name]
            if name in vel_map:
                self.dq_current[i] = vel_map[name]
        if not self._initialized:
            self.controller.reset(self.q_current.copy())
            self._initialized = True

    def _wrench_cb(self, msg: WrenchStamped):
        """接收 F/T 传感器回调."""
        self.wrench_raw[0] = msg.wrench.force.x
        self.wrench_raw[1] = msg.wrench.force.y
        self.wrench_raw[2] = msg.wrench.force.z
        self.wrench_raw[3] = msg.wrench.torque.x
        self.wrench_raw[4] = msg.wrench.torque.y
        self.wrench_raw[5] = msg.wrench.torque.z

    def _control_loop(self):
        """主控制循环 (定时触发)."""
        if not self._initialized:
            return

        # 1) 摩擦补偿
        tau_compensated = self.sim_to_real.friction_compensation(
            self.wrench_raw, self.dq_current
        )

        # 2) 控制步
        q_cmd = self.controller.step(
            self.q_current, tau_compensated, obs=self.obs_current
        )

        # 3) 延迟补偿
        q_cmd = self.sim_to_real.latency_compensation(q_cmd)

        # 4) 发布关节指令
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q_cmd.tolist()
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        self._pub_joint_cmd.publish(msg)

    def set_policy_action(self, action: np.ndarray):
        """外部设置策略动作 (来自 /policy_action 订阅)."""
        if action.shape[0] >= 7:
            self.obs_current = action  # 假设 action 包含完整观测
