# 开门评估报告
## session/routeB-cartesian-vic — 2026-06-22

---

## 一、现有策略评估

### 对比表 (30 episodes each, current env with obs_adapt)

| 模型 | 开门率 | 平均K | 平均door | 最大接触力 | 风格 |
|:----|:-----:|:-----:|:--------:|:---------:|:----|
| **L0_real_seed0** | **46.7%** (14/30) | 717 | 0.237 rad | 293N | 💪 蛮力硬推 |
| VIC_PPO_seed0 | 3.3% (1/30) | 184 | 0.069 rad | 91N | 🧘 低K柔顺 |

### 关键发现

1. **L0_real_seed0 能开门** (47% SR) 但 K=717 = 硬推，接触力最高 293N
2. **VIC_PPO_seed0 是真正的柔顺策略** (K=184) 但 env 漂移导致 3% SR
3. 所有现有模型训练时 obs_dim=33，当前 env 有未提交的 damping 维 → obs_dim=34 → 需要 obs_adapt

### VIC_PPO_seed0 的 env 漂移链

从训练到现在的环境变更：
```
8c672806 Paper2 SEG-3 (训练时 env)
  ↓ gravity comp / DOF fix / stiffness clamp
81da4c45
  ↓ contact_force fix
59e6cfb3
  ↓ EFF_MASS 2.0→1.0
d073396a
  ↓ bias + entropy fix
65aca678
  ↓ P1 QACC clamp (MAX_TAU=500)
2057f58e
  ↓ 未提交: K_EXPLORE→K_LAZY, damping dim, LAMBDA_F ↑
CURRENT ──→ 漂移过大，旧策略失效
```

## 二、现有架构评估

### 已完成

| 组件 | 状态 | 测试 |
|:----|:----:|:----:|
| CompliantController + Cartesian VIC | ✅ | E2E 6/6 |
| LoadedPolicy (SB3状态字典兼容) | ✅ | 加载VIC_PPO_seed0 OK |
| vic_policy_to_cartesian() 映射 | ✅ | [-1,1] → 物理笛卡尔值 |
| YAML config_loader | ✅ | door_opening.yaml 加载OK |
| 重力补偿 (MuJoCo/FK) | ✅ | 代码完成 |
| Jacobian 三套实现 | ✅ | MuJoCo/KDL/Approx |
| 导纳四相位 | ✅ | VariableAdmittance |

### 未完成

| 组件 | 状态 | 原因 |
|:----|:----:|:-----|
| RL策略 → CompliantController 真联通 | ❌ | 所有策略obs_dim=33 vs env obs_dim=34 |
| 真机测试 | ❌ | 无硬件 |
| 接触力可视化 | ❌ | 需要跑MuJoCo完整物理step |

## 三、下一步推荐

### 选项A: 对齐env + 重新训练 (推荐)
1. **提交当前改动** — K_EXPLORE→K_LAZY_COEF, damping dim, LAMBDA_F
2. 用新 env 重新训练 PPO (300k steps)
3. 产出的新策略 obs_dim=34，与 env 完全对齐
4. 用 CompliantController 加载新策略跑全链路测试

### 选项B: 直接用 L0_real_seed0 跑全链路
1. L0_real_seed0 有 47% SR（虽不柔顺但能开门）
2. 通过 CompliantController 的导纳层可额外压减接触力
3. 适合快速验证从策略到真机指令的全链路

### 选项C: 修复 VIC_PPO_seed0 兼容性
1. 临时 revert obs_dim 到 33（去掉 damping 维）
2. VIC_PPO_seed0 K=184 是真柔顺
3. 但长期看重新训练更干净
