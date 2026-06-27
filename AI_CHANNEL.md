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
