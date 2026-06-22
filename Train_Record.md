# 仿真开门 — 训练记录
## session/routeB-cartesian-vic → 2026-06-22

---

## 环境状态

| 项目 | 值 | 说明 |
|:----|:---:|:-----|
| env | `cartesian_vic_env.py` | obs_dim=34, K_LAZY_COEF, LAMBDA_F=0.002, r_force ungated |
| commit | `80fbd3e1` | 已提交至父仓库 paper2-cartesian-vic |
| train | `train_curriculum.py` | entropy 0.10→0.005, train_damping=True |
| 旧模型 | 全部 obs_dim=33 | 与新 env 不兼容，需重新训练 |

## 冒烟测试 (30k steps CPU)

```
算法: PPO (4 envs, lr=3e-4, batch=256, ent_coef=0.05)
用时: 28s (CPU)
结果:
  K: mean=528 [50, 1000] — 无坍塌
  door: max=0.3175 rad — 已超阈值 0.3rad
  NaN: 无
  K collapse: 无
```

→ **奖励修复验证成功。** 新 reward 方向正确，训练有效。

## 300k 正式训练命令

```bash
cd xarm7_cartesian_vic

# 单种子 (seed=0, 约20min on GPU)
python train_curriculum.py --algo PPO --curriculum-levels 1 \
    --level-steps 300000 --eval-episodes 30 --seed 0 \
    --run-name aligned_v1_seed0 --device cuda

# 三种子 (seed 0/1/2, 约1h on GPU)
# 分别跑:
#   python train_curriculum.py ... --seed 0 --run-name aligned_v1_seed0
#   python train_curriculum.py ... --seed 1 --run-name aligned_v1_seed1
#   python train_curriculum.py ... --seed 2 --run-name aligned_v1_seed2
```

## 预期结果 (基于30k冒烟 + 旧模型47%基线)

| 指标 | 旧L0_real_seed0 | 新模型(预期) |
|:----|:--------------:|:-----------:|
| 开门率 | 47% | ≥50% |
| 平均K | 717 (硬推) | **≤300 (柔顺)** |
| 接触力 | 最高293N | **≤50N** |
| obs_dim | 33 | 34 (对齐) |

## CompliantController 集成

完成后跑:
```bash
# 导出策略
python train_curriculum.py ... --export-policy

# 用 CompliantController 加载并可视化
python -m xarm7_compliant.examples.demo_mujoco --vic \
    --policy runs/aligned_v1_seed0/level_0/policy.pt \
    --scene ../mujoco_rl/door_real_scene.xml
```
