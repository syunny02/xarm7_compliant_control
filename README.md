# xArm7 柔性控制包 — Sim-to-Real Compliant Control

> **打造者**: 强化学习之父 Davied  
> **目标**: 为 xArm7 开门任务提供可在真机复现的柔性控制

---

## 目录

1. [核心原理](#1-核心原理)
2. [项目结构](#2-项目结构)
3. [快速开始: 仿真验证](#3-快速开始-仿真验证)
4. [真机部署指南](#4-真机部署指南)
5. [控制循环详解](#5-控制循环详解)
6. [调参策略](#6-调参策略)
7. [与现有系统的关系](#7-与现有系统的关系)

---

## 1. 核心原理

### 1.1 什么是柔性控制？

柔性控制 (Compliant Control) 让机器人具有"主动性顺从"能力。当外部力作用时，机器人不是刚性抵抗，而是顺应外力方向做微调，从而:

- **安全**: 与人/环境碰撞时降低冲击力
- **自适应**: 自动补偿位置误差 (如手柄抓握偏差)
- **鲁棒**: 对模型不精确不敏感

### 1.2 导纳控制 (本包采用)

```
M · Δq̈ + D · Δq̇ + K · Δq = τ_ext
```

| 符号 | 含义 | 作用 |
|------|------|------|
| M | 虚拟质量 | 惯性 — 越大响应越慢越平稳 |
| D | 虚拟阻尼 | 振荡抑制 — 越大运动越黏滞 |
| K | 虚拟刚度 | 位置保持 — 越大越"硬" |
| τ_ext | 外部关节力矩 | 来自 F/T 传感器 × Jacobian 转置 |
| Δq | 位置偏移 | 最终输出比期望位置偏移的量 |

### 1.3 变量导纳 (本包特色)

**开门任务四相位**:

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3
 接近门      抓握手柄     拉门        完成
 K=80        K=40        K=15        K=60
 (刚性)      (中等)      (柔性)      (恢复)
```

高接触力 → 低刚度 (越推越让)  
低接触力 → 高刚度 (走直线)

---

## 2. 项目结构

```
xarm7_compliant_control/
├── README.md                          ← 你现在看的
├── setup.py                           ← pip 安装
├── requirements.txt                   ← 依赖
├── config/
│   ├── default.yaml                   ← 通用配置
│   └── door_opening.yaml              ← 开门专用配置
└── xarm7_compliant/
    ├── __init__.py
    ├── controller.py                  ← 主流水线 (核心)
    ├── sim_to_real.py                 ← 仿真→真机桥梁
    ├── core/
    │   ├── __init__.py
    │   ├── admittance.py              ← 导纳滤波器 (数学核心)
    │   ├── policy.py                  ← 策略加载器 (兼容 .pt)
    │   └── wrench.py                  ← F/T → 关节力矩映射
    ├── ros/
    │   ├── __init__.py
    │   └── node.py                    ← ROS2 真机节点
    ├── examples/
    │   ├── demo_mujoco.py             ← MuJoCo 仿真验证
    │   └── demo_real_robot.py         ← 真机测试入口
    └── tests/
        ├── test_admittance.py         ← 导纳单元测试
        └── test_wrench.py             ← 力映射单元测试
```

---

## 3. 快速开始: 仿真验证

### 3.1 安装

```bash
cd xarm7_compliant_control
pip install -e .                # 安装核心
pip install -e .[sim]           # 安装 + 仿真依赖
```

### 3.2 运行导纳单元测试

```bash
python -m xarm7_compliant.tests.test_admittance
```

预期输出:
```
[PASS] test_zero_force
[PASS] test_steady_state_offset: dq=0.1234 (≈0.1250)
[PASS] test_max_delta_limit: max|Δq|=0.0500 ≤ 0.05
[PASS] test_variable_admittance: Phase0 Δq=0.0012 < Phase2 Δq=0.0432
[PASS] test_reset

✓ 所有导纳滤波器测试通过
```

### 3.3 运行 MuJoCo 仿真 Demo

```bash
python -m xarm7_compliant.examples.demo_mujoco
```

这个 demo 会:
1. 创建 7-DOF 机器人 (可加载真实 door_scene.xml)
2. 施加 4 种外力和: 空闲 → 20N 恒力 → 交变力 → 脉冲
3. 观察导纳滤波器的位移响应
4. 绘制力-位移-力矩曲线图

---

## 4. 真机部署指南

### 4.1 ROS2 节点

```bash
# 启动真机柔性控制节点
ros2 run xarm7_compliant compliant_node \
  --ros-args -p policy_path:="/home/ub/models/best_multimodal.pt" \
             -p use_sim:=false \
             -p control_freq:=100.0 \
             -p admittance_damping:=15.0 \
             -p admittance_stiffness:=30.0
```

### 4.2 真机调参流程

| 步骤 | 操作 | 预期现象 |
|------|------|----------|
| 1 | 设置 K=80, D=20, max_delta=0.03 | 高刚性, 几乎无偏移 |
| 2 | 手动轻推末端 | 应感到硬阻力, 位置几乎不动 |
| 3 | 设置 K=15, D=10, max_delta=0.15 | 明显柔性 |
| 4 | 手动推拉 | 应感到弹性跟随, 放手后回到原轨迹 |
| 5 | 逐步减小 D 直到感觉轻微振荡 | 找到临界阻尼 |
| 6 | D = 临界 × 1.5 | 最优阻尼 |

### 4.3 安全操作

> 🚨 **首次真机运行时，请遵循:**
> 1. 先用 `--use-sim` 标志确认参数安全
> 2. 从高阻尼 (D=25)、低刚度 (K=Fz×0.5)、小偏移 (max_delta=0.03) 开始
> 3. 始终将手放在 E-stop 上
> 4. 监控 `compliant_status` topic 的 ||τ|| 值，超过 30 Nm 应立即停止

---

## 5. 控制循环详解

```
                 ┌─────────────────────────────────────────┐
 传感器           │           控制流水线                      │       执行器
 ┌─────┐         │  ┌──────┐    ┌──────────┐    ┌───────┐  │       ┌──────┐
 │关节  │──q,dq──→│  │策略   │──→│导纳滤波器 │──→│安全   │──│──q_cmd→│xArm7 │
 │编码器│         │  │推理   │   │MΔq̈+DΔq̇+K│   │监控器 │  │       │关节  │
 └─────┘         │  └──────┘    │=τ_ext     │   └───────┘  │       └──────┘
 ┌─────┐         │              └─────┬─────┘              │
 │F/T  │──→wrench────→τ_ext=Jᵀ·w──→──┘                    │
 │传感器│         │                                         │
 └─────┘         └─────────────────────────────────────────┘
```

### 每一步的事件顺序:

```
t=0ms    读取 /joint_states (q, dq) 和 F/T (wrench_raw)
t=2ms    偏置补偿 + 重力补偿 + 低通滤波
t=3ms    Jacobian 转置: τ_ext = J(q)^T · wrench  
t=4ms    策略推理 (如果加载了 .pt 文件)
t=6ms    导纳积分: q_cmd = q_des + Δq  
t=8ms    安全检查: 限位/限速/紧急停止
t=10ms   发布 /compliant_joint_cmd
```

---

## 6. 调参策略

### 6.1 参数速查表

| 你想 ... | 调大 → | 调小 → | 优先级 |
|----------|--------|--------|--------|
| 更顺服外力 | K↑, D↓, M↑ | K↓, D↑, M↓ | 1. K 2. D 3. M |
| 减少振荡 | D↑ | D↓ | 1. D 2. M |
| 允许大偏移 | max_delta↑ | max_delta↓ | 1 |
| 响应更快 | M↓ | M↑ | 1. M 2. D |
| 更安全 | max_delta↓, D↑ | max_delta↑, D↓ | 1. 限位 |

### 6.2 开门任务推荐参数

| 任务阶段 | M | D | K | max_delta |
|----------|---|---|---|-----------|
| 初始位置 → 接近门 | 0.5 | 20 | 80 | 0.03 |
| 靠近手柄 → 抓握 | 1.0 | 15 | 40 | 0.08 |
| 抓握 → 拉门 | 1.5 | 10 | 15 | 0.15 |
| 门开 → 保持 | 0.5 | 20 | 60 | 0.05 |

### 6.3 Sim-to-Real 适配

```python
from xarm7_compliant.sim_to_real import SimToRealConfig, SimToRealWrapper

st = SimToRealWrapper(SimToRealConfig(
    coulomb_friction=0.15,     # 真机额外摩擦补偿
    viscous_friction=0.05,
    control_latency_s=0.02,    # 延迟补偿
))

# 真机循环中:
tau_comp = st.friction_compensation(tau_raw, dq)  # 摩擦补偿
q_cmd = st.latency_compensation(q_cmd)              # 延迟补偿
```

---

## 7. 与现有系统的关系

本包与项目原有代码的关系:

```
┌─────────────────────────────────────────────────────────┐
│                    xarm7_door_ros2                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │ mujoco_rl/                                       │    │
│  │   ├── train_her.py       (PPO/SAC 训练)          │    │
│  │   ├── door_env.py        (MuJoCo 环境)           │    │
│  │   └── standalone_compliant_controller.py (旧)     │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ xarm7_compliant_control/      ← **你在这里**     │    │
│  │   ├── 模块化设计, pip 可安装                     │    │
│  │   ├── 同时支持仿真 + 真机 ROS2                  │    │
│  │   └── 变量导纳 + 安全监控 + Sim-to-Real         │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ src/xarm7_door_policy/   (ROS2 策略部署)         │    │
│  │   ├── policy_loader.py                           │    │
│  │   ├── admittance_controller_node.py (旧 ROS2)    │    │
│  │   └── task_supervisor_node.py                    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**向后兼容**: 
- 本包可直接加载 `mujoco_rl/runs/*/best_model.pt` 策略
- 本包的 WrenchMapper 兼容 ROS2 `geometry_msgs/WrenchStamped`
- 本包的 ROS2 节点与 `xarm7_door_interfaces` 兼容

---

## 附录 A: 数学推导

### A.1 一维导纳

```
M·ẍ + D·ẋ + K·x = f
```

稳态: 当 ẍ=0, ẋ=0 时, `x = f/K`  
即恒外力下偏移量与刚度成反比。

### A.2 离散化 (半隐式 Euler)

```
a_k = (f_k - D·v_k - K·x_k) / M    ← 加速度
v_{k+1} = v_k + a_k · dt            ← 更新速度
x_{k+1} = x_k + v_{k+1} · dt        ← 更新位置
```

---

## 附录 B: 常见问题

**Q: 真机上手柄抓不准怎么办？**  
A: 降低 Phase 2 的刚度 K (如 5-10 N/m), 让机器人主动顺应手柄位置。

**Q: 关门时力太大怎么办？**  
A: 降低 `max_force_n` 到 50N, 加大滤波 `filter_alpha=0.4`。

**Q: 策略输出的位置有抖动？**  
A: 增大导纳的阻尼 D, 或在策略输出后加低通滤波器。

---

> _Sim-to-Real 不是转移, 是适应。_
> — Davied, 强化学习之父
