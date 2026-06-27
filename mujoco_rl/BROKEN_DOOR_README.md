# Broken-Door Suite — 受损门参数化场景 (Route B 第二篇核心环境)

> 作者: AI-Assistant (SoniXChat). 对标 Adroit door_broken_joint_{easy,medium,hard} (OffDynamicsRL/ODRL).
> 设计原则: 损坏必须 **可参数化 / 可控 / 可证伪**. 已在 mujoco 3.9.0 本机验证.

## 文件
| 文件 | 说明 |
|---|---|
| door_real_scene.xml | baseline 正常门 (hinge: damping=1, frictionloss=0, stiffness=0) |
| door_broken_hinge_easy.xml | 轻度受损 |
| door_broken_hinge_medium.xml | 中度受损 |
| door_broken_hinge_hard.xml | 重度受损 |

## 损坏建模 (全部作用在 hinge 关节, 三个正交物理维度)
| 维度 | 物理含义 | baseline | easy | medium | hard |
|---|---|---|---|---|---|
| damping | 铰链生锈/卡涩 (速度阻尼) | 1 | 5 | 20 | 60 |
| frictionloss | 干摩擦死区 (须克服才动) | 0 | 2 | 8 | 20 |
| stiffness+springref | 门歪斜/自动回弹 (弹簧偏置) | 0 | 2 / -0.15 | 6 / -0.35 | 15 / -0.60 |

## 可证伪性验证 (本机实测, _falsify.py)
对所有门施加 **相同恒定开门力矩 -8 N*m**, 测最大开门角:

| 门 | 最大开门角 (rad) | 解读 |
|---|---|---|
| baseline | -1.846 (撞满量程) | 正常门, 轻松开 |
| easy | -1.176 | 明显变涩 |
| medium | -0.105 | 几乎卡死 |
| hard | -0.059 | 基本打不开 |

**结论: MONOTONIC = True (越坏越难开, 严格单调).**
=> 同一刚性策略 baseline 能开 / hard 开不了 => 必须靠 VIC 顺应 (调刚度+加力) => 柔性价值可量化证明.

## 论文用法建议 (选项, B 定)
1. **课程学习**: baseline -> easy -> medium -> hard, 配合 env 已有 curriculum_level.
2. **off-dynamics 视角**: 训练在某档, 测试在另一档, 对标 ODRL benchmark (源/目标动力学不一致).
3. **核心论点**: 固定刚度策略在 hard 失效 (frictionloss 死区顶不过) -> 可变阻抗策略学会 "先加力破静摩擦再顺应" -> 这就是第二篇要证的 "柔性=安全且有效".

## 注意 (诚实标注)
- medium/hard 的难度主要由 **frictionloss 干摩擦死区** 主导: -8N*m 顶不过 8/20 的静摩擦, 门纹丝不动.
  这比纯加阻尼更接近 "真实卡死", 但也意味着策略必须主动加大力 —— 训练时注意 K_MAX/力矩饱和上限要够.
- 损坏数值是对标 Adroit 量级 + 本机证伪测试调出来的, 不是 xarm7 真实门测量值. 上真机阶段需按实物重标.
