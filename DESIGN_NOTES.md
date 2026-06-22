# xArm7 柔性控制包 — 设计与实现说明

> **作者**: 强化学习之父 Davied  
> **日期**: 2026-06-08 (初版) / 2026-06-12 (路线B重构)  
> **分支**: `compliant-control`  
> **基分支**: `mujoco-rl-stage1`

---

## 版本历史

### v2.0 — 路线B: Cartesian VIC 集成 (2026-06-12)

**动机**: CompliantController 原假设策略输出关节位置，但 VIC 训练策略输出笛卡尔增量 [dx, K, gripper]，两者接口不匹配。

**改动清单**:

| # | 文件 | 改动 |
|:-:|:-----|:------|
| 1 | `controller.py` | `step()` 新增 `vic_action` 参数 + `_get_jacobian()` 内部差分IK + `last_gripper` 透传 |
| 2 | `core/jacobian_provider.py` | **新建** — MuJoCo (mj_jacSite) / KDL (ChainJntToJac) / 近似 三套实现 |
| 3 | `core/wrench.py` | 重力补偿实现 (MuJoCo用qfrc_bias, 真机用正运动学+末端质量/CoM) + 删除废弃的ROS2JacobianProvider |
| 4 | `core/admittance.py` | `AdmittanceConfig.stiffness_from_policy` + `set_stiffness()` 方法，策略K可直接覆盖导纳K |
| 5 | `config_loader.py` | **新建** — 递归 YAML → dataclass 填充，支持嵌套结构 |
| 6 | `__init__.py` | 导出新模块: `CompliantController`, `CompliantControllerConfig`, `load_config` |
| 7 | `setup.py` / `requirements.txt` | 添加 `pyyaml>=6.0` 依赖 |
| 8 | `tests/test_e2e_with_policy.py` | **新建** — 6项全链路 E2E 测试 |
| 9 | `examples/demo_mujoco.py` | 新增 `run_demo_vic()` + `--vic` 参数，展示 Cartesian VIC 全链路 |

**新架构**:
```
VIC policy → [dx(6), K, gripper]
    ↓
controller.step(q, wrench, vic_action=...)
  1. J = get_jacobian(q)
  2. dq = J_pinv @ dx_cart       ← 差分IK
  3. q_policy = q + dq           ← 关节目标
  4. admittance.set_stiffness(K) ← 策略刚度覆盖
  5. admittance.update(q_policy, tau_ext) → q_cmd
  6. 限速/限位/急停
```

**测试结果**: 6/6 通过
- ✅ 零力无漂移
- ✅ 恒外力导纳响应
- ✅ 策略刚度覆盖 (低K→大偏移, 高K→小偏移)
- ✅ 夹爪指令透传
- ✅ 笛卡尔 IK 有效
- ✅ 安全限幅生效

---

## 一、概述

`xarm7_compliant_control` 是一个面向 **xArm7 机械臂** 的模块化柔性控制包，支持从 **MuJoCo 仿真** 到 **真机部署** 的完整流程。核心思想是 **Sim-to-Real**：在仿真中验证控制算法，通过参数平滑迁移到真机。

### 设计原则

1. **模块化** — 导纳、策略、力映射各自独立，可替换
2. **可验证** — 每个模块都有单元测试，端到端仿真验证
3. **可部署** — 提供 ROS2 真机节点，零修改即可运行
4. **安全优先** — SafetyMonitor 监控超限自动急停

---

## 二、包结构

```
xarm7_compliant_control/
├── config/                          ← 配置文件（YAML）
│   ├── default.yaml                 ← 通用配置
│   └── door_opening.yaml            ← 开门任务专用
├── xarm7_compliant/                 ← Python 包
│   ├── __init__.py                  ← 导出 CompliantController
│   ├── controller.py                ★ 主流水线
│   ├── sim_to_real.py               ★ Sim-to-Real 桥梁
│   ├── core/
│   │   ├── __init__.py
│   │   ├── admittance.py            ★ 导纳滤波器
│   │   ├── policy.py                ★ 策略加载器
│   │   └── wrench.py                ★ 力/力矩映射
│   ├── ros/
│   │   ├── __init__.py
│   │   └── node.py                  ★ ROS2 真机节点
│   ├── examples/
│   │   ├── demo_mujoco.py           ★ 仿真验证
│   │   └── demo_real_robot.py       ★ 真机测试
│   └── tests/
│       ├── test_admittance.py       ✓ 导纳测试（5项）
│       ├── test_wrench.py           ✓ 力映射测试（3项）
│       ├── test_imports.py          ✓ 导入测试
│       ├── test_demo_import.py      ✓ Demo导入测试
│       └── test_e2e_mujoco.py       ✓ 端到端仿真
├── setup.py                         ← pip 安装
├── requirements.txt                 ← 依赖列表
└── README.md                        ← 使用文档
```

---

## 三、核心模块详解

### 3.1 导纳控制器 (`core/admittance.py`)

**AdmittanceFilter** — 关节空间导纳滤波器

```
tau_ext → [导纳方程] → q_offset → q_cmd = q_desired + q_offset
```

二阶导纳方程：
```
M * d²(Δq) + D * d(Δq) + K * Δq = tau_ext
```

- `M` (惯性): 控制响应速度
- `D` (阻尼): 控制震荡抑制
- `K` (刚度): 控制位置刚度

**VariableAdmittance** — 可变导纳（四相位）
- Phase 0: 低刚度 (安装) → 0.3x
- Phase 1: 中等刚度 (操作) → 1.0x
- Phase 2: 高刚度 (抓取) → 3.0x  
- Phase 3: 最高刚度 (牵引) → 5.0x

支持相位间线性插值，避免刚度突变。

### 3.2 策略加载器 (`core/policy.py`)

**LoadedPolicy** — 加载训练好的 RL 策略

```
obs → [Policy Network] → action (关节位置增量)
```

- 支持 `.pt` / `.pth` / `.zip` 格式
- 自动检测输入/输出维度
- 推理模式 (torch.no_grad)
- CPU/GPU 自动选择

**ZeroPolicy / ConstantPolicy** — 用于调试的虚拟策略

### 3.3 力/力矩映射 (`core/wrench.py`)

**WrenchMapper** — 六维力/力矩 → 关节力矩

```
F_ext (6D) → Jacobian转置 → tau_ext (7D)
```

- `tau_ext = J(q)^T * F_ext`
- 一阶低通滤波 (α = 0.3)
- 安全钳位 (单关节 ≤ 5 Nm)
- 力传感器偏置补偿

### 3.4 主流水线 (`controller.py`)

**CompliantController** — 编排所有模块

```
q_desired
    ↓
[RL Policy] → q_action + [Admittance] → q_cmd
    ↑                                    ↓
obs ← [MuJoCo 仿真 / 真机] ← q_cmd + tau_ext
```

```python
ctrl = CompliantController(config)
q_cmd = ctrl.step(q_desired, wrench, obs=obs)
```

**SafetyMonitor** — 安全监控
- 关节位置超限 → WARN
- 关节速度超速 → STOP
- 关节力矩超限 → STOP
- 力/力矩超限 → STOP

### 3.5 Sim-to-Real 桥梁 (`sim_to_real.py`)

**SimToRealWrapper** — 参数平滑过渡

```
sim_params = [D_sim, M_sim, K_sim, alpha_sim]
    ↓ (线性插值，e.g. 100步)
real_params = [D_real, M_real, K_real, alpha_real]
```

- 默认 100 步过渡 (可配置)
- 每步更新导纳参数 + 滤波系数
- 避免突然切换导致的震荡

### 3.6 ROS2 真机节点 (`ros/node.py`)

**CompliantControlNode** — ROS2 节点

```
订阅:
  /joint_states          → 当前关节状态
  /force_torque          → 六维力数据
  /compliant/cmd         → 目标位置command
  
发布:
  /compliant/joint_cmd   → 柔性控制输出的关节指令
  /compliant/status      → 控制器状态
  /compliant/safety      → 安全事件
```

启动:
```bash
ros2 run xarm7_compliant compliant_node --ros-args -p config:=config/door_opening.yaml
```

---

## 四、测试结果

| 测试 | 状态 | 描述 |
|------|------|------|
| `test_admittance.py` | ✅ 5/5 | 零力、稳态偏移、限幅、可变导纳、重置 |
| `test_wrench.py` | ✅ 3/3 | 滤波、偏置、安全钳位 |
| `test_imports.py` | ✅ | 完整流水线导入 |
| `test_demo_import.py` | ✅ | MuJoCo demo 导入 |
| `test_e2e_mujoco.py` | ✅ | 端到端仿真验证 |

### 端到端仿真验证细节

在 MuJoCo 7-DOF 手臂模型上：

| 场景 | 结果 |
|------|------|
| 零外力 | Δq ≈ 0 (无漂移) |
| Fx=20N (稳态) | 稳态偏移 ≈ 0.03 rad |
| Fx=50N (脉冲) | 峰值偏移 ≈ 0.10 rad，恢复后回到零 |
| SafetyMonitor | 超限时触发紧急停止 |

---

## 五、使用流程

### 5.1 本地安装

```bash
cd xarm7_compliant_control
pip install -e .          # 仅核心
pip install -e .[sim]     # 含 MuJoCo 仿真
pip install -e .[real]    # 含 ROS2 真机
```

### 5.2 仿真验证

```bash
python -m xarm7_compliant.examples.demo_mujoco
# 或
python -m xarm7_compliant.tests.test_e2e_mujoco
```

### 5.3 真机部署

```bash
# 方式1: 直接运行
python -m xarm7_compliant.examples.demo_real_robot

# 方式2: ROS2 节点
ros2 run xarm7_compliant compliant_node
```

### 5.4 配置自定义

编辑 `config/default.yaml` 或传入自定义配置：

```python
from xarm7_compliant import CompliantController
with open("my_config.yaml") as f:
    ctrl = CompliantController(f.read())
```

---

## 六、Sim-to-Real 检查清单

从仿真迁移到真机前，请确认：

- [ ] 导纳参数已通过 SimToRealWrapper 平滑过渡
- [ ] 力传感器已标定（偏置补偿）
- [ ] Jacobian 矩阵已验证（与真机运动学一致）
- [ ] SafetyMonitor 限幅值已根据真机调整
- [ ] 策略输入 obs 维度与真机一致
- [ ] 通信频率匹配（仿真 200Hz vs 真机 100Hz）
- [ ] 实机首次运行时设置 `safety.emergency_stop=true`

---

## 七、与已有 xArm7 工作流的集成

```
mujoco_rl/                               ← 训练环境
    └── train_policy.py → best.pth
            ↓ (策略路径)
xarm7_compliant_control/                 ← 部署包
    ├── config/default.yaml
    │     ├── policy.path: "../mujoco_rl/runs/best.pth"
    │     └── sim_to_real.enabled: true
    └── ros/node.py → 真机执行
```

训练好的 `.pth` 策略文件只需配置路径，无需修改代码即可在真机运行。

---

## 八、版本信息

| 组件 | 版本 |
|------|------|
| Python | 3.12+ |
| numpy | ≥1.24 |
| PyTorch | ≥2.0 |
| MuJoCo | ≥3.1 (可选) |
| ROS2 | Humble (可选) |
| xArm SDK | ≥2.0 (可选) |

---

*本文档由 Davied（强化学习之父）编写，记录 xarm7_compliant_control 包的设计思路与实现细节。*
