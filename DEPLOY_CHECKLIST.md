# 真机部署准备清单
## 现状 → 目标：真机一到就能跑

---

## 一、策略：多模态61维 (已训练好的模型)

### 现状
旧 multimodal 模型因 env 漂移不可用，需要重新训练。

### 待办
- [ ] 重新训练 multimodal 策略 (PPO, 100万步, ~2h GPU)
- [ ] 验证 29-dim (无视觉) 和 61-dim (有视觉) 两个版本
- [ ] 导出 ROS2 兼容的 `policy.pt`

### 训练命令
```bash
cd mujoco_rl
python train_multimodal.py --algo PPO --steps 1000000 \
    --seed 0 --run-name multimodal_v2_novis_s0 --device cuda

python train_multimodal.py --algo PPO --steps 1000000 \
    --seed 0 --run-name multimodal_v2_vis_s0 --use-visual --device cuda
```

---

## 二、观测管道 (已就绪，无需改动)

| 数据 | 仿真来源 | 真机来源 | 状态 |
|:----|:--------|:--------|:----|
| 关节位置(7) | MuJoCo qpos | `/joint_states` | ✅ |
| 关节速度(7) | MuJoCo qvel | `/joint_states` | ✅ |
| TCP位姿(7) | FK计算 | FK计算 | ✅ |
|  wrench(6)  | MuJoCo接触力 | `/uf_ftsensor_raw_states` | ✅ topic已有 |
| 视觉特征(32) | 仿真渲染→MobileNetV3 | RealSense→visual_encoder_node | ✅ 节点已有 |
| 任务编码(2) | env内部 | gripper状态推断 | ✅ |

---

## 三、控制管道 (CompliantController)

| 组件 | 文件 | 状态 | 真机时需要 |
|:----|:----|:----|:----------|
| 导纳控制器 | `core/admittance.py` | ✅ 完整 | 无需改动 |
| F/T→力矩映射 | `core/wrench.py` | ✅ 有重力补偿 | 标定 bias + ee_mass |
| Jacobian | `core/jacobian_provider.py` | ✅ KDL占位 | 实现 `KDLJacobianProvider` |
| 策略加载 | `core/policy.py` | ✅ SB3兼容 | 无需改动 |
| ROS2节点 | `ros/node.py` | ✅ 框架 | 参数调优 |
| Sim2Real适配 | `sim_to_real.py` | ✅ 完整 | 标定 friction/delay |
| YAML配置 | `config/door_opening.yaml` | ✅ 完整 | 按需调参 |

---

## 四、真机标定流程 (真机到后做)

```
[1h] F/T传感器标定
  └─ ros2 run xarm7_compliant compliant_node --calibrate-ft
  └─ 采集静止数据 → bias_wrench()
  └─ 设 ee_mass / ee_com 参数

[30min] 重力补偿验证
  └─ 移动机器人到不同姿态
  └─ 确认补偿后 wrench ≈ 0

[30min] 摩擦系数标定
  └─ 空跑关节轨迹 → 测 friction_coulomb / viscous_friction

[1h] 安全测试
  └─ 低刚度模式 (K=10) 手推机器人 → 确认导纳响应
  └─ 急停测试 → 确认 emergency_stop_tau 触发

[30min] KDL Jacobian 部署
  └─ 从 robot_description 参数加载 URDF
  └─ 验证 6×7 Jacobian 输出
```

---

## 五、启动命令 (真机到后)

```bash
# 终端1: 启动 xArm 驱动
ros2 launch xarm_api xarm7_driver.launch.py robot_ip:=192.168.1.117

# 终端2: 启动视觉编码器 (如果用视觉)
ros2 run xarm7_door_perception visual_encoder_node

# 终端3: 启动柔性控制
ros2 run xarm7_compliant compliant_node \
    --ros-args -p policy_path:=/path/to/multimodal_v2_policy.pt \
               -p use_sim:=false \
               -p admittance_stiffness:=40.0
```

---

## 六、当前就绪度

| 组件 | 就绪度 | 还需要什么 |
|:----|:-----:|:----------|
| 策略 (61-dim multimodal) | 0% | **重新训练 (~2h GPU)** |
| 观测管道 | 90% | 视觉编码器需验证 |
| 导纳控制器 | 95% | KDL Jacobian 实现 |
| ROS2节点 | 85% | 真机参数标定 |
| Sim2Real | 80% | 真机标定数据 |
| **总计** | **~60%** | **缺策略 + KDL实现** |
