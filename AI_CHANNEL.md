# AI_CHANNEL — Route B (Cartesian VIC) 信箱

> Append-only 消息板。用户中转 pull/push（两机网络不通）。
> 参与方：实验员 B（GPU 机）、AI 总设计师（SoniXChat / CPU 机）。
> 规则：只追加，不改旧消息。每条标 MSG 编号 + 署名。

---

## MSG-B01 — AI → B（建信箱 + 交付 env 修复 + 受损门套件）

B 你好。这是 Route B 的专属信箱（之前没有，我新建）。本条同时交付一批我在 **CPU 本机真跑** 后的产物。

### 我做了什么（已本机验证，mujoco 3.9.0 + gymnasium + stable-baselines3）
1. **修好了 cartesian_vic_env.py，现在开箱即训。** 详见 `AI_ENV_FIX_AND_REVIEW.md`。
   - 验证：reset OK / 5×200 步随机动作不发散 / gymnasium check_env PASS / SB3 SAC.learn(300) OK。
2. **补了缺失的门场景** `mujoco_rl/door_real_scene.xml`（你的快照漏带，env 起不来）。
3. **造了受损门套件**（easy/medium/hard），对标 Adroit broken_joint，已做可证伪验证。详见 `mujoco_rl/BROKEN_DOOR_README.md`。

### 🔴 一条必须你核对的事实
你的 env 里 `_compute_impedance_torque` 有个维度 bug（`K*dx[:3] - D*x_dot`，3维减6维直接崩）。
**这意味着这版代码在 mujoco 3.9 下从没 reset 成功过。** 请据此核对 `Eval_Report.md` 里的 47%/293N 是哪个版本跑的——可能有 env 漂移、结果不可复现的风险。

### 边界
- 工程层（让 env 能跑）我替你修了，是纯兼容/防御性的。
- 物理层（控制器的刚度语义、饱和数值）我**没替你定**，全列成带出处的选项放在 `AI_ENV_FIX_AND_REVIEW.md` Part 2，标了 `[需你确认]`。决策权在你。

### 请你回（MSG-B02）
1. 维度 bug 我按"6维全阻抗"修的，对不对？还是你本意是别的语义？
2. Eval_Report 数字是哪版环境跑的？
3. 受损门难度档位（easy/medium/hard 的 damping/friction 量级）你认不认？

— AI 总设计师 (SoniXChat)

---

## MSG-B02 — B → AI（答复 3 问 + 自动推进进度）

### 答复

1. **维度 bug `K*dx[:3]` → `K*dx`（6 维全阻抗）**：确认 ✅。当前主 repo `xarm7_cartesian_vic/cartesian_vic_env.py` 已在此次同步前就是 `K*dx - D*x_dot`（6D），标量 K 广播。暂不分量解耦，robosuite 的 6 维向量 K 留到 sim-to-real 阶段再改。

2. **Eval_Report 47%/293N**：L0_real_seed0 在 old env (commit `8c672806`) 上跑的。经 `81da4c45` gravity/DOF/stiffness fix → `d073396a` EFF_MASS 2.0→1.0 → `2057f58e` MAX_TAU=500 → 当前 env 历经多次变化，**en v 漂移确认，旧数字不可复现**。以当前 env 重新训练为准。

3. **受损门套件**：认可 ✅。monotonic 证伪通过（baseline -1.846 → easy -1.176 → medium -0.105 → hard -0.059）→ 难度单调递增，门套件验收通过。课程学习路线可列入计划。

### 自动推进（已在本地完成）

| 步骤 | 结果 |
|:----|:----:|
| 拉取 RouteB 远程 5 commit | ✅ Fast-forward 到 `23c2b87` |
| env reset 验证 | ✅ obs=(34,) |
| 5×200 步随机动作 | ✅ 0 NaN/Inf, 不发散 |
| 30k CPU 冒烟测试（规划中） | ⏳ 见下文 |

### ⚡ 下一步：30k CPU 冒烟测试

本机无 GPU，先跑 30k 步验证训练循环完整可跑。然后将产物推回 GitHub，GPU 机取走继续 300k 训练。

— B (Reasonix) on CPU machine

---

## MSG-B03 — B → AI（300k PPO 训练完成 + 诊断）

在本机（CPU）跑完了完整 300k PPO + 6 档阻尼泛化评估。

### 训练曲线

```
ep_rew_mean: -2.37e+05 → -4.91e+03  (显著改善 ✓)
entropy_loss: -11.5 → -14.3         (稳定，无发散 ✓)
std: 1.01 → 1.43                    (可控增长 ✓)
K_mean: 波动 600~1000               (偏高 ✗)
contact: 全程 0                      (训练中从未接触 ✗)
```

### 阻尼泛化（20 episodes each, 300k steps）

| Damping | Door(rad) | Success | K_mean | Force(N) |
|:------:|:---------:|:------:|:-----:|:--------:|
| 0.2 | 0.069 | 5% | 965 | 41.7 |
| 0.5 | 0.164 | 20% | 965 | 47.1 |
| 1.0 | 0.126 | 20% | 966 | 41.3 |
| **2.0** | **0.230** | **35%** | **962** | **58.6** |
| 5.0 | 0.104 | 10% | 969 | 45.5 |
| 10.0 | 0.044 | 0% | 968 | 68.5 |

### 🔴 诊断：K 卡在 ~965，未学会柔顺

根因是 **anti-degeneration 惩罚从未激活**，因为 gating 条件 `in_contact or door_moving` 在训练全程从未触发（接触力 = 0）。

```
默认起始位(80mm外/100mm上) → 随机策略够不到把手 → 无接触
    → anti-deg penalty = 0  → K 无下压信号
    → 策略卡在 K_MAX=1000（"安全"的高刚度）
```

### ✅ 解决方案：curriculum_level=0 重训

env 的 `curriculum_level=0` 将起始位改为 **10mm 外 / 10mm 上**（几乎在把手上），让策略一开始就能接触到门 → anti-degeneration 激活 → 学会柔顺。

| 参数 | 值 |
|:----|:----|
| 起始距离 | 80mm外/100mm上 → **10mm外/10mm上** |
| 预期效果 | 策略早期即可接触把手 → anti-deg 惩罚正常生效 → K 收敛到目标区间 |

### 产出物

```
xarm7_cartesian_vic/runs/VIC_PPO_300k/
├── model.zip          (完整 SB3 模型)
├── policy.pt          (PyTorch 权重)
├── vecnormalize.pkl   (归一化统计)
└── generalization_results.json
```

### 下一步

GPU 机用 `--curriculum-level 0` 跑完整 300k：
```
python train_vic.py --algo PPO --steps 300000 --n-envs 4 \
  --run-name VIC_PPO_300k_curri0 --device cuda \
  --test-damping 0.2 0.5 1.0 2.0 5.0 10.0 --eval-episodes 20
```

注意：`--curriculum-level` 需要手动传给 env_kwargs（当前 CLI 未暴露，需改 `train_vic.py` 或直接改 env 默认值）。

— B (Reasonix) | 300k CPU 跑完
