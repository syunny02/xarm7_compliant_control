# AI Env Fix & Design Review — Route B (Cartesian VIC)

> 作者: AI-Assistant (SoniXChat). 在 CPU 本机用 mujoco 3.9.0 / gymnasium / stable-baselines3
> 真跑过你的 env 后整理. 分两部分:
> **Part 1 = 工程修复与验证事实 (已做, 可复现).**
> **Part 2 = 控制器设计建议 (建议, 决策权在 B).**
> 凡是我拍的数值或语义假设, 都标了 [需你确认]. 我没有替你定物理灵魂.

---

## Part 1 — 工程修复 (让 env 能从起不来到开箱即训)

### 验证结果 (本机实测)
| 项 | 结果 |
|---|---|
| reset + 30 步零动作 | OK, 无 NaN |
| 5 episodes x 200 步随机动作 (模拟 RL 早期乱探索) | nonfinite_obs = 0, 不发散 |
| gymnasium.utils.env_checker.check_env | PASS |
| stable_baselines3 SAC("MlpPolicy").learn(300) | OK, 训练循环跑通 |

### 修复清单 (5 处, 都是 "真跑" 才暴露的)
| # | 位置 | 问题 | 修法 | 性质 |
|---|---|---|---|---|
| 1 | env 启动 | 缺 door_real_scene.xml (在 GPU 机有, 本机没有) | 用第一篇几何造了 baseline 门, 统一成你的命名 (hinge / latch_joint / handle_grip / link_tcp) | 补缺失 |
| 2 | env L69-74 | mju_mat2Vel 在 mujoco 3.9 已移除 | shim: mju_mat2Quat + mju_quat2Vel 两步替代 | 纯 API 兼容 |
| 3 | env L235 | F_imp = K*dx[:3] - D*x_dot -> 3维 减 6维 崩溃 | 改 K*dx (走 6 维) | **[需你确认] 见 Part2-A** |
| 4 | env L233-236 | 力矩无饱和导致 QACC NaN/Inf 数值发散 | wrench/torque 饱和 (F_MAX=150, TAU_MAX=100) | **[需你确认] 数值是拍的** |
| 5 | env L322 | _x_des += dx 无界累加 -> 目标点漂到不可达 | 限制相对初始 TCP 偏移 (XDES_MAX_OFFSET=0.25m) | **[需你确认] 见 Part2-B** |

### 关键 一个必须你核对的事实
第 3 个维度 bug 意味着: **这版 VIC 在当前 mujoco 3.9 下连 reset 都跑不通** (改之前直接维度不匹配崩).
所以你 Eval_Report.md 里那些数字 (47% / 293N) 是哪个版本/哪套环境跑的 —— 需要你核对, 有 env 漂移、结果不可复现的风险.

### F_MAX / TAU_MAX / XDES_MAX_OFFSET 说明
这三个常量是我为了让仿真 "不发散" 临时拍的量级, **不是从 xarm7 真实规格推出来的**.
上真机前必须按 xarm7 关节力矩上限 (datasheet) 和工作空间重标. 现在的作用是 "让仿真能跑", 不是 "物理正确".

---

## Part 2 — 设计建议 (带出处, 你来定)

我调研了 3 个先进项目, 对照你的 env 各修复点. 都是建议, 不是结论.

### 参考项目
- SERL (rail-berkeley/serl_franka_controllers, 230*) — Berkeley RL 笛卡尔阻抗部署
- robosuite OSC (ARISE-Initiative/robosuite, 2482*) — 操作型 RL 黄金标准
- Cartesian-Impedance-Controller (matthias-mayr, 330*, JOSS DOI:10.21105/joss.05194) — 工业级 C++

### A. [需你确认] 维度修法: 全 K vs 6 维分量解耦
- 现状: 我把标量 K 广播到 6 维了. 但 3 维 bug 原意可能是想区分位置/姿态 (才取 shape).
- robosuite 做法: kp 是 **6 维向量** (位置姿态分开) + **uncouple_pos_ori=True** (位置/姿态控制律解耦).
- 含义: 如果你本意是位置姿态用不同 K 刚度, 应该走分量向量而不是 "标量广播到六维".
- 建议 (可选): K 拆 6 维, pos/ori 分开. **但这是物理语义决策, 你说了算, 我没改.**

### B. [印证] 目标位姿限幅 = SERL 论文核心机制, 方向应该对
- 我加的 XDES_MAX_OFFSET 限幅, 担心是不是乱搞. 调研发现这正是 SERL 论文写的:
  > "limiting the reference point to be within a certain distance from the current pose...
  >  a high gain can be used for accuracy without excess force when in contact."
- 含义: 限制参考点偏移是 **接触安全 + 高增益不爆力** 的标准 VIC 手段 (不是我乱加的).
- 建议: 把这个限幅作为正式设计写进 DESIGN_NOTES 的安全机制, 数值再按你的真机重标.

### C. [建议] 饱和限幅 -> 加 jerk (加加速度) 限制 + 分量
- 我现在只 clip 力/力矩, 比较糙. 真机 clip 会引入抖动.
- 工业级 (matthias-mayr) 做法: **jerk limitation + 对 stiffness/pose/wrench 分别独立限幅**.
- 建议: 这是 sim-to-real 阶段的事. 现在能跑 (粗暴饱和) 够用, 上真机时再细化.

### D. [建议] 训练时参考 robosuite 的 variable_kp 模式
- robosuite 的 impedance_mode 有三档: fixed / variable / variable_kp.
- **variable_kp (策略只调刚度, 阻尼自动算) = 你这条路线的对应物.**
- 建议: 你的策略输出阻抗参数, 正好对标 benchmark, "策略学刚度, 阻尼临界" 是可引用的标准做法, 论文里好写.

---

## 一句话总结
Part 1 我修到能开箱即训了 (有验证, 是 env 修复).
Part 2 是建议, 我没碰会影响你训练结果 / 100% 论点的物理灵魂. 凡 [需你确认] 的, 决策权都在你 (我标了出处).
