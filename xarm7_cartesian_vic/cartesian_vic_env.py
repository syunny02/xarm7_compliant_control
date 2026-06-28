#!/usr/bin/env python3
"""
Cartesian Variable Impedance Control (VIC) environment for xArm7 door opening.
Built on MuJoCo physics with Gymnasium API.

── Key Innovation ──
Instead of joint-space position control (Paper 1), the policy outputs:
1) Cartesian delta pose (dx, dy, dz, droll, dpitch, dyaw)
2) Variable stiffness K ∈ [50, 1000] N/m
3) Gripper command

The impedance control law computes joint torques:
τ = Jᵀ · [K · (x_des - x) - D · ẋ] + gravity_comp
where D = 2·√(K·m_eff) is critically damped.

Gravity compensation uses data.qfrc_bias applied via qfrc_applied.
Actuator model: gainprm→0 (neutralized), torque via qfrc_applied.

── Anti-Degeneration ──
Two regularization terms prevent the policy from reverting to stiff position control:
- Stiffness regularization: L_K = λ_K · (K/1000)² (lazy stiffness penalized)
- Contact force penalty: L_F = λ_F · max(0, ‖F_ext‖ - 30)²

── Damping Generalization ──
Train on one door hinge damping (default=1), test zero-shot on:
{0.2, 0.5, 1, 2, 5, 10}
by modifying model.dof_damping[dof_id] at reset() time.

── Curriculum Learning (MSG 057-058) ──
curriculum_level parameter controls approach distance to handle:
- 0: 10mm outside, 10mm above (hand almost on handle)
- 1: 30mm outside, 40mm above
- 2: 50mm outside, 70mm above
- 3: 80mm outside, 100mm above (original)
- None: backward-compatible default (80/100mm)

── Action Space (Box, 8 dims, continuous, normalized [-1,1]) ──
[0:3] Cartesian delta position [-1,1] → [-0.05, 0.05] m
[3:6] Cartesian delta orientation [-1,1] → [-0.1, 0.1] rad (axis-angle)
[6] Stiffness K [-1,1] → [0,1] → [50, 1000] N/m
[7] Gripper [-1,1] → [0, 255]

── Observation Space (Box, 34 dims) ──
[0:7] Joint positions
[7:14] Joint velocities
[14:17] TCP position (world frame)
[17:20] TCP orientation (axis-angle)
[20:23] TCP linear velocity
[23:26] TCP angular velocity
[26] Current stiffness K (for policy state awareness)
[27] Door hinge damping (domain randomization)
[28] Handle angle
[29] Door angle
[30] Distance to handle grip
[31:34] Relative vector TCP → handle grip
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "..", "mujoco_rl", "door_real_scene.xml")


class CartesianVICEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    K_MIN = 50.0
    K_MAX = 1000.0
    DELTA_POS_MAX = 0.05
    DELTA_ROT_MAX = 0.1
    DAMPING_RATIO = 1.0
    EFF_MASS_DEFAULT = 1.0

    # ── Anti-degeneration (MSG36 verified) ──
    # ── Anti-degeneration (MSG36 V3c verified → K≈115, 100% SR) ──
    LAMBDA_K_HIGH = 0.001  # stiffness regularization weight (K > K_TARGET, gated)
    K_TARGET = 500.0       # target stiffness
    LAMBDA_K_LOW = 0.2     # low-K penalty, ALWAYS active (prevents K collapse)
    K_LOW_THRESH = 200.0   # K floor
    LAMBDA_F = 0.0005      # contact force penalty weight (gated)
    FORCE_THRESHOLD = 30.0 # N, soft threshold
    REACH_GATE = 0.15
    W_REACH = 4.0
    W_HANDLE = 1.5
    W_DOOR_DELTA = 5.0    # ⬆ V3c verified: double V3a → better door movement incentive
    W_DOOR_ABS = 0.5
    CTRL_COST = 0.005
    SUCCESS_BONUS = 500.0   # ⬇ V3c verified: half of current → focus on incremental reward

    def __init__(self, xml_path=DEFAULT_XML, max_steps=300, frame_skip=5,
                 render_mode=None, render_size=(720, 960),
                 door_open_threshold=0.3, test_damping=None,
                 exploration_steps=0, curriculum_level=None, train_damping=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_dof_adrs = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                dof_start = self.model.jnt_dofadr[jid]
                if jid + 1 < self.model.njnt:
                    dof_count = self.model.jnt_dofadr[jid + 1] - dof_start
                else:
                    dof_count = self.model.nv - dof_start
                for d in range(dof_count):
                    self.arm_dof_adrs.append(dof_start + d)
        self.arm_dof_adrs = np.array(sorted(self.arm_dof_adrs), dtype=int)
        self.nv_arm = len(self.arm_dof_adrs)

        for i in range(7):
            self.model.actuator_gainprm[i, 0] = 0.0
            self.model.actuator_biasprm[i, :] = [0.0] * self.model.actuator_biasprm.shape[1]

        self.sid_tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self.sid_grip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "handle_grip")
        self.jid_door = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self.jid_handle = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "latch_joint")
        self.qadr_door = self.model.jnt_qposadr[self.jid_door]
        self.qadr_handle = self.model.jnt_qposadr[self.jid_handle]
        self.dof_door = self.model.jnt_dofadr[self.jid_door]

        self.body_tcp = self.model.site_bodyid[self.sid_tcp]

        self.home_ctrl = None
        if self.model.nkey > 0:
            self.home_ctrl = self.model.key_ctrl[0].copy()

        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self._render_size = render_size
        self._renderer = None
        self._cam = None
        self.door_open_threshold = float(door_open_threshold)
        self.test_damping = test_damping
        self.exploration_steps = int(exploration_steps)
        self.train_damping = train_damping
        self.curriculum_level = curriculum_level

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        obs_dim = 7 + 7 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        self._x_des = np.zeros(6, dtype=np.float64)
        self._current_K = self.K_MIN
        self._current_damping = 1.0
        self.cur_step = 0
        self._prev_phi = 0.0
        self._max_door_ang = 0.0
        self._contact_force_hist = []
        self._last_contact_step = -100

    def _get_tcp_pose(self):
        pos = self.data.site_xpos[self.sid_tcp].copy()
        mat = self.data.site_xmat[self.sid_tcp].reshape(3, 3).copy().ravel()
        quat = np.zeros(4)
        rot = np.zeros(3)
        mujoco.mju_mat2Quat(quat, mat)
        mujoco.mju_quat2Vel(rot, quat, 1.0)
        return np.concatenate([pos, rot])

    def _get_tcp_vel(self):
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.sid_tcp)
        v_lin = jac_pos @ self.data.qvel
        v_ang = jac_rot @ self.data.qvel
        return np.concatenate([v_lin, v_ang])

    def _get_obs(self):
        tcp_pose = self._get_tcp_pose()
        tcp_vel = self._get_tcp_vel()
        grip_pos = self.data.site_xpos[self.sid_grip]
        dist = float(np.linalg.norm(grip_pos - tcp_pose[:3]))
        rel_vec = grip_pos - tcp_pose[:3]
        return np.concatenate([
            self.data.qpos[:7].copy(),
            self.data.qvel[:7].copy(),
            tcp_pose,
            tcp_vel,
            np.array([self._current_K]),
            np.array([self._current_damping]),
            np.array([self.data.qpos[self.qadr_handle]]),
            np.array([abs(self.data.qpos[self.qadr_door])]),
            np.array([dist]),
            rel_vec,
        ]).astype(np.float64)

    def _potential(self, dist, handle_angle):
        return -self.W_REACH * dist - self.W_HANDLE * abs(handle_angle)

    def _compute_impedance_torque(self, x_des, K):
        tcp_pos = self.data.site_xpos[self.sid_tcp]
        tcp_rot = np.zeros(3)
        _mat = self.data.site_xmat[self.sid_tcp].reshape(3, 3).copy().ravel()
        _quat = np.zeros(4)
        mujoco.mju_mat2Quat(_quat, _mat)
        mujoco.mju_quat2Vel(tcp_rot, _quat, 1.0)
        x_cur = np.concatenate([tcp_pos, tcp_rot])
        dx = x_des - x_cur
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.sid_tcp)
        J_full = np.vstack([jac_pos, jac_rot])
        J_arm = J_full[:, self.arm_dof_adrs]
        v_arm = self.data.qvel[self.arm_dof_adrs]
        D = 2.0 * np.sqrt(K * self.EFF_MASS_DEFAULT)
        x_dot = J_arm @ v_arm
        F_imp = K * dx - D * x_dot  # Full 6D impedance  # Per-DOF 6D impedance  # Full 6D positional + rotational impedance
        tau_arm = J_arm.T @ F_imp
        return tau_arm, F_imp

    def _get_contact_force(self):
        total_force = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geom1 = self.model.geom_bodyid[c.geom1]
            geom2 = self.model.geom_bodyid[c.geom2]
            if geom1 != self.body_tcp and geom2 != self.body_tcp:
                continue
            force_local = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force_local)
            single_force = float(np.linalg.norm(force_local[0:3]))
            if c.dist < 0 and single_force > 100.0:
                continue
            frame_33 = self.data.contact[i].frame.reshape(3, 3)
            cf_world = frame_33.T.copy() @ force_local[:3]
            R = self.data.site_xmat[self.sid_tcp].reshape(3, 3).copy()
            total_force[:3] += R.T @ cf_world
            total_force[3:] += np.cross(c.pos - self.data.site_xpos[self.sid_tcp], R.T @ cf_world)
        return float(np.linalg.norm(total_force[:3])), total_force[:3].copy()

    def _compute_reward(self, info):
        dist = info["dist_to_grip"]
        handle = info["handle_angle"]
        door = info["door_ang"]
        force_norm = info["contact_force"]
        K = info["K"]

        phi = self._potential(dist, handle)
        r_shape = phi - self._prev_phi
        self._prev_phi = phi

        r_door_delta = self.W_DOOR_DELTA * door
        r_door_abs = self.W_DOOR_ABS * door if dist < self.REACH_GATE else 0.0
        r_success = self.SUCCESS_BONUS if info["success"] else 0.0

        # ── Anti-degeneration penalties (MSG36 dual threshold) ──
        in_contact = force_norm > 0.001
        door_moving = door > 0.05  # V3c: 0.02, raised slightly to avoid init noise

        # Low-K penalty: ALWAYS active, penalize K < K_LOW_THRESH
        r_K_low = -self.LAMBDA_K_LOW * max(0.0, self.K_LOW_THRESH - K) ** 2

        # High-K penalty + contact force: only when in contact or door moving
        r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2
        r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2

        r_ctrl = -self.CTRL_COST

        reward = r_shape + r_door_delta + r_door_abs + r_success + r_force + r_K_low + r_K_high + r_ctrl
        return reward, {
            "r_shape": r_shape, "r_door_delta": r_door_delta,
            "r_door_abs": r_door_abs, "r_success": r_success,
            "r_force": r_force, "r_K_low": r_K_low, "r_K_high": r_K_high, "r_ctrl": r_ctrl,
        }

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} out of bounds"
        act = np.clip(action, -1.0, 1.0).astype(np.float64)

        dx = act[0:3] * self.DELTA_POS_MAX
        drot = act[3:6] * self.DELTA_ROT_MAX
        K_raw = (act[6] + 1.0) / 2.0
        K = float(self.K_MIN + K_raw * (self.K_MAX - self.K_MIN))
        gripper_cmd = int(np.clip((act[7] + 1.0) / 2.0 * 255, 0, 255))

        if self.exploration_steps > 0 and self.cur_step < self.exploration_steps:
            K = max(K, 800.0)

        K = float(np.clip(K, self.K_MIN, self.K_MAX))
        self._current_K = K
        self._x_des[:3] += dx
        self._x_des[3:] += drot

        if self.home_ctrl is not None:
            self.data.ctrl[:] = self.home_ctrl
        self.data.ctrl[7] = gripper_cmd

        tau_arm, F_imp = self._compute_impedance_torque(self._x_des, K)
        # P1: Clamp applied torques to prevent QACC instability
        MAX_TAU = 500.0  # N-m per joint
        tau_arm = np.clip(tau_arm, -MAX_TAU, MAX_TAU)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.arm_dof_adrs] = tau_arm
        self.data.qfrc_applied[self.arm_dof_adrs] += self.data.qfrc_bias[self.arm_dof_adrs]

        physics_ok = True
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)):
                physics_ok = False
                break
        if physics_ok:
            mujoco.mj_forward(self.model, self.data)

        self.cur_step += 1
        force_norm, _ = self._get_contact_force()
        door_ang = float(abs(self.data.qpos[self.qadr_door]))
        self._contact_force_hist.append(force_norm)
        self._max_door_ang = max(self._max_door_ang, door_ang)
        handle_angle = float(self.data.qpos[self.qadr_handle])
        grip_pos = self.data.site_xpos[self.sid_grip]
        tcp_pos = self.data.site_xpos[self.sid_tcp]
        dist = float(np.linalg.norm(grip_pos - tcp_pos))
        contact_success = force_norm > 3.0 or (self.cur_step - self._last_contact_step < 10)
        success = (door_ang >= self.door_open_threshold) and contact_success
        if force_norm > 3.0:
            self._last_contact_step = self.cur_step

        info = {
            "door_ang": door_ang,
            "handle_angle": handle_angle,
            "dist_to_grip": dist,
            "contact_force": force_norm,
            "K": K,
            "success": success,
        }

        reward, rew_parts = self._compute_reward(info)
        info.update({f"reward_{k}": v for k, v in rew_parts.items()})

        terminated = success
        truncated = self.cur_step >= self.max_steps
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:7] += self.np_random.uniform(-0.03, 0.03, size=7)

        # 从 options 读取 test_damping / train_damping (允许覆盖构造参数)
        opt = options or {}
        _test_damping = opt.get("test_damping", self.test_damping)
        _train_damping = opt.get("train_damping", self.train_damping)

        self._x_des = self._get_tcp_pose()
        tcp_pos_init = self._x_des.copy()
        grip_pos_init = self.data.site_xpos[self.sid_grip].copy()

        # 先把 TCP 升到把手高度 + 水平推向把手
        target_pos = grip_pos_init[:3].copy()
        target_pos[2] += 0.03  # 把手正上方 3cm
        for _ in range(100):
            diff = target_pos - self._x_des[:3]
            if np.linalg.norm(diff) < 0.01:
                break
            step = np.clip(diff, -0.02, 0.02)
            self._x_des[:3] += step
            tau_arm, _ = self._compute_impedance_torque(self._x_des, 600.0)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.arm_dof_adrs] = tau_arm
            self.data.qfrc_applied[self.arm_dof_adrs] += self.data.qfrc_bias[self.arm_dof_adrs]
            mujoco.mj_step(self.model, self.data)

        if self.curriculum_level is not None:
            offsets = [(0.05, 0.03), (0.10, 0.05), (0.15, 0.08), (0.20, 0.12)]
            lvl = min(self.curriculum_level, len(offsets) - 1)
            xy_off, z_off = offsets[lvl]
        else:
            xy_off, z_off = 0.08, 0.10

        grip_pos = self.data.site_xpos[self.sid_grip].copy()
        approach_pos = grip_pos[:3].copy()
        # 从把手后退 xy_off 米、抬高 z_off 米
        push_dir = grip_pos[:3] - self.data.site_xpos[self.sid_tcp][:3]
        push_dir[2] = 0.0
        pn = float(np.linalg.norm(push_dir))
        if pn > 0.001: push_dir /= pn
        else: push_dir = np.array([1.0, 0.0, 0.0])
        approach_pos[:2] -= push_dir[:2] * xy_off
        approach_pos[2] = grip_pos[2] + z_off

        for _ in range(100):
            diff = approach_pos - self._x_des[:3]
            if np.linalg.norm(diff) < 0.002:
                break
            step = np.clip(diff, -0.01, 0.01)
            self._x_des[:3] += step
            tau_arm, _ = self._compute_impedance_torque(self._x_des, 400.0)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.arm_dof_adrs] = tau_arm
            self.data.qfrc_applied[self.arm_dof_adrs] += self.data.qfrc_bias[self.arm_dof_adrs]
            mujoco.mj_step(self.model, self.data)

        self._x_des[:3] = approach_pos
        self._x_des[3:] = self._get_tcp_pose()[3:]
        self._current_K = self.K_MIN
        self._current_damping = 1.0

        if _test_damping is not None:
            self._current_damping = float(_test_damping)
            self.model.dof_damping[self.dof_door] = self._current_damping
        elif _train_damping:
            self._current_damping = float(np.exp(self.np_random.uniform(np.log(0.2), np.log(10.0))))
            self.model.dof_damping[self.dof_door] = self._current_damping
        else:
            self._current_damping = 1.0

        if self.home_ctrl is not None:
            self.data.ctrl[:] = self.home_ctrl

        mujoco.mj_forward(self.model, self.data)
        self.cur_step = 0
        tcp_pos = self.data.site_xpos[self.sid_tcp]
        grip_pos = self.data.site_xpos[self.sid_grip]
        dist0 = float(np.linalg.norm(grip_pos - tcp_pos))
        handle0 = float(self.data.qpos[self.qadr_handle])
        self._prev_phi = self._potential(dist0, handle0)
        self._max_door_ang = float(abs(self.data.qpos[self.qadr_door]))
        self._contact_force_hist = []
        self._last_contact_step = -100
        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._render_size[1], self._render_size[0])
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0.3, 0.0, 0.4]
            self._cam.distance = 1.5
            self._cam.azimuth = 135
            self._cam.elevation = -30
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
