# xArm7 Compliant Door Opening — 完整架构方案
## 目标：仿真训练 → 真机部署，零修改

---

## 一、核心设计原则

> **"仿真怎么控制，真机就怎么控制"**

xArm7 真机接受**位置指令**（ros2_control position interface），
所以仿真也必须用**位置控制**来训练策略，而不是当前的阻抗控制（力矩）。

## 二、架构

```
┌─────────────────────────────────────────────────────────┐
│                   统一控制层                              │
│                                                          │
│  RL策略 → [dx(3), drot(3), K(1), gripper(1)]            │
│      ↓                                                   │
│  IK: dq = J_pinv @ [dx, drot]                            │
│      ↓                                                   │
│  q_des = q_cur + dq   (关节目标位置)                     │
│      ↓                                                   │
│  F/T传感器 → τ_ext = Jᵀ · F                              │
│      ↓                                                   │
│  导纳: q_cmd = q_des + Δq(τ_ext)  ← 柔顺偏移             │
│      ↓                                                   │
│  安全限幅 → 关节位置指令 → xArm7                          │
│                                                          │
│  ┌────────────── 仿真 ──────────────┐ ┌─── 真机 ──────┐  │
│  │ MuJoCo: mj_step(q_cmd)         │ │ ros2_control  │  │
│  │ mj_forward → 物理碰撞 → F      │ │ 物理世界 → F/T│  │
│  └──────────────────────────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**关键：仿真和真机用同一套控制代码。** 唯一的区别是物理引擎。

## 三、改造方案

### 3.1 仿真环境改造：从阻抗控制 → 位置控制 + 导纳

**当前 (阻抗控制)**:
```
env.step(action):
  tau = Jᵀ · K · dx - D · v     ← 力矩控制
  qfrc_applied = tau + gravity   ← MuJoCo发力矩
  mj_step()                      ← 物理步进
```

**改后 (位置控制 + 导纳)**:
```
env.step(action):
  dx = action[:3] * 0.05
  K = 50 + (action[6]+1)/2 * 950
  q_des = q + J_pinv @ dx        ← IK
  τ_ext = Jᵀ · F_contact         ← 读接触力
  q_cmd = q_des + adm_update(q, τ_ext, K)  ← 导纳偏移
  data.qpos[:7] = q_cmd          ← 直接设位置
  mj_forward(m, d)               ← 仅运动学更新
  F_contact = read_contact()     ← 读碰撞力
```

这样训练出来的策略，**直接输出就是真机要的关节位置指令**。

### 3.2 观测空间：只用真机能拿到的信息

| 观测维度 | 当前(34维) | **改后(真机兼容)** | 来源 |
|:--------|:---------|:-----------------|:-----|
| 关节位置(7) | ✅ | ✅ | joint_states |
| 关节速度(7) | ✅ | ✅ | joint_states |
| TCP位姿(6) | ✅ | ✅ | FK计算 |
| TCP速度(3) | ✅ | ✅ | FK计算 |
| 刚度K(1) | ✅ | ✅ | 策略内部状态 |
| 门阻尼(1) | ✅ | ❌ 去掉 | 真机没有 |
| 把手角度(1) | ✅ | ❌ 去掉 | 需要视觉 |
| 门角度(1) | ✅ | ❌ 去掉 | 需要视觉 |
| 到把手距离(1) | ✅ | ❌ 去掉 | 需要视觉 |
| TCP→把手向量(3) | ✅ | ❌ 去掉 | 需要视觉 |
| **FT力觉(6)** | ❌ | **✅ 新增** | F/T传感器 |
| **任务编码(2)** | ❌ | **✅ 新增** | 相位/状态 |

**改后观测: 7+7+6+3+1+6+2 = 32维**，全部真机可拿。

### 3.3 奖励函数：综合考虑所有真实因素

```python
# 开门奖励
r_door = 20.0 * door_angle            # 门开越大越好

# 安全约束 (硬约束, 惩罚极大)
r_force = -0.001 * max(0, F - 30)²    # 接触力 > 30N 惩罚
r_torque = -0.0001 * max(0, τ - 15)²  # 关节力矩 > 15Nm 惩罚

# 柔顺约束
r_K_low = -0.2 * max(0, 200 - K)²     # K < 200 惩罚 (防坍塌)
r_K_high = -0.001 * max(0, K - 500)²  # K > 500 惩罚 (防蛮力, gated)

# 效率约束
r_ctrl = -0.005 * ||action||²         # 动作平滑
r_time = -1.0 / max_steps             # 时间惩罚 (尽快开门)

# 稀疏奖励  
r_success = +50.0 if door_open        # 开门成功

reward = r_door + r_force + r_torque + r_K_low + r_K_high + r_ctrl + r_time + r_success
```

### 3.4 Sim-to-Real 层

```python
class Sim2RealAdapter:
    """仿真→真机适配器 (部署时启用)"""
    
    def adapt_obs(self, obs_sim):
        """仿真obs → 真机obs (去掉仿真特有维度)"""
        # 34维 → 32维: 去掉门阻尼(27), 把手角度(28→27), ...
        pass
    
    def adapt_wrench(self, wrench_raw):
        """真机F/T → 仿真格式 (去偏置+重力补偿+滤波)"""
        w = wrench_raw - self.bias
        w = self.gravity_comp(q, w)
        w = lowpass_filter(w)
        return w
    
    def friction_comp(self, tau_raw, dq):
        """摩擦补偿 (真机摩擦 > 仿真)"""
        return tau_raw + sign(dq)*0.15 + dq*0.05
```

## 四、训练部署流程

```
Step 1: 改造仿真环境 (位置控制+导纳)
        用时: ~2h
        产出: compliant_env.py (obs=32维)
        
Step 2: 训练策略 (TQC, 300k-600k步)
        用时: GPU 30-60min
        产出: policy.pt
        
Step 3: 真机部署
        用时: ~1h
        产出: ros2 run xarm7_compliant compliant_node
              --policy runs/tqc_final/policy.pt
              --use-sim false

Step 4: 真机标定
        用时: ~30min
        - F/T 零点偏置: bias_wrench()
        - 重力补偿参数: ee_mass, ee_com
        - 摩擦系数: coulomb_friction, viscous_friction
```

## 五、当前进度 vs 需要做的

| 组件 | 当前状态 | 需要改什么 |
|:----|:--------|:----------|
| 导纳控制器 | ✅ 完整 | 无需改动 |
| Jacobian | ✅ MuJoCo/KDL/近似 | KDL需要真机部署时实现 |
| 重力补偿 | ✅ qfrc_bias/FK | 真机标定参数 |
| 仿真环境 | ❌ 阻抗控制 | **改位置控制+导纳** |
| 观测空间 | ❌ 34维含门信息 | **改32维真机可用** |
| 奖励函数 | ✅ MSG36双阈值 | 小调整 |
| TQC训练 | ✅ 可运行 | 需要再跑一轮 |
| ROS2节点 | ✅ 框架 | 参数微调 |
