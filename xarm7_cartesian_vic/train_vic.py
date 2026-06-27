#!/usr/bin/env python3
"""Train a Cartesian VIC door-opening policy (PPO / SAC / A2C) on MuJoCo.

Extends Paper 1's train_mujoco.py with:
- Cartesian VIC environment (torque-controlled impedance)
- Anti-degeneration monitoring (K, contact force logged to TensorBoard)
- Damping generalization: --test-damping for zero-shot eval at end of training
- Comparison mode: --baseline to run joint-space (Paper 1) for ablation

Outputs (under runs/<run-name>/):
model.zip Full SB3 model
policy.pt Plain PyTorch policy weights
tb/ TensorBoard logs

Examples:
python train_vic.py --algo PPO --steps 300000
python train_vic.py --algo PPO --steps 300000 --test-damping 0.2 0.5 2 5 10
python train_vic.py --algo SAC --steps 500000 --seed 1
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import torch

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from cartesian_vic_env import CartesianVICEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MUJOCO_RL = os.path.join(HERE, "..", "mujoco_rl")

ALGOS = {"PPO": PPO, "SAC": SAC, "A2C": A2C}
ONPOLICY = {"PPO", "A2C"}


class VICProgressCallback(BaseCallback):
    """Log VIC-specific metrics (stiffness, contact force, damping generalization)
    to TensorBoard for monitoring anti-degeneration."""
    def __init__(self, log_every=10000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self._next = log_every

    def _on_step(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        Ks = [i.get("K", 0.0) for i in infos if isinstance(i, dict)]
        forces = [i.get("contact_force", 0.0) for i in infos if isinstance(i, dict)]
        doors = [i.get("door_ang", 0.0) for i in infos if isinstance(i, dict)]
        succ = [1.0 if i.get("success") else 0.0 for i in infos if isinstance(i, dict)]
        lk = [i.get("L_K", 0.0) for i in infos if isinstance(i, dict)]
        lf = [i.get("L_F", 0.0) for i in infos if isinstance(i, dict)]

        if Ks:
            self.logger.record("vic/K_mean", float(np.mean(Ks)))
            self.logger.record("vic/K_max", float(np.max(Ks)))
        if forces:
            self.logger.record("vic/contact_force_mean", float(np.mean(forces)))
            self.logger.record("vic/contact_force_max", float(np.max(forces)))
        if doors:
            self.logger.record("door/mean_ang", float(np.mean(doors)))
            self.logger.record("door/max_ang", float(np.max(doors)))
        if succ:
            self.logger.record("door/success_rate", float(np.mean(succ)))
        if lk:
            self.logger.record("vic/L_K_reg", float(np.mean(lk)))
        if lf:
            self.logger.record("vic/L_F_reg", float(np.mean(lf)))

        if self.num_timesteps >= self._next:
            self._next += self.log_every
            msg = (f"[{self.num_timesteps:>8}] door={float(np.mean(doors)) if doors else 0:.3f} "
                   f"K={float(np.mean(Ks)) if Ks else 0:.0f} "
                   f"F={float(np.mean(forces)) if forces else 0:.1f}")
            print(msg)
        return True


class EntropyAnnealingCallback(BaseCallback):
    """Anneal ent_coef from start_ent_coef down to end_ent_coef over total_timesteps.

    Implements entropy scheduling (David Silver L9):
    - Early high entropy encourages exploration to discover contact-rich behaviors.
    - Late low entropy lets the policy converge to a stable door-opening strategy.
    """
    def __init__(self, start_ent_coef=0.05, end_ent_coef=0.005, total_timesteps=300000, verbose=0):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps
        self._print_next = 10000

    def _on_step(self):
        ratio = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.model.ent_coef = self.start_ent_coef - (self.start_ent_coef - self.end_ent_coef) * ratio

        if self.num_timesteps >= self._print_next:
            self._print_next += 10000
            print(f"[entropy annealing] timestep={self.num_timesteps:>8} ent_coef={self.model.ent_coef:.6f}")
        return True


def evaluate_damping_generalization(model, dampings, vn_path=None, n_episodes=10, max_steps=300):
    """Zero-shot evaluation across door hinge damping values.
    If vn_path is provided, wraps the eval env in VecNormalize (loaded from training stats)
    with training=False, norm_reward=False for proper inference (MSG 038 §4-③).
    Returns dict {damping: {mean_door_ang, success_rate, mean_K, mean_force}}."""
    results = {}
    for damping in dampings:
        eval_env = CartesianVICEnv(
            max_steps=max_steps,
            test_damping=damping,
            render_mode=None,
        )
        # Wrap in VecNormalize for proper eval (MSG 038 §4-③)
        from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv
        venv = _DummyVecEnv([lambda: eval_env])
        if vn_path and os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, venv)
            venv.training = False
            venv.norm_reward = False
        door_angs = []
        successes = []
        Ks = []
        forces = []
        for ep in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_door = 0.0
            ep_success = False
            ep_Ks = []
            ep_forces = []
            while not done:
                # VecNormalize expects batched obs; use raw env obs for predict
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                done = term or trunc
                ep_door = max(ep_door, info.get("door_ang", 0.0))
                if info.get("success"):
                    ep_success = True
                ep_Ks.append(info.get("K", 0.0))
                ep_forces.append(info.get("contact_force", 0.0))
            door_angs.append(ep_door)
            successes.append(1.0 if ep_success else 0.0)
            Ks.append(float(np.mean(ep_Ks)) if ep_Ks else 0.0)
            forces.append(float(np.max(ep_forces)) if ep_forces else 0.0)
        results[f"damping_{damping}"] = {
            "damping": damping,
            "mean_door_ang": float(np.mean(door_angs)),
            "std_door_ang": float(np.std(door_angs)),
            "success_rate": float(np.mean(successes)),
            "mean_K": float(np.mean(Ks)),
            "mean_contact_force": float(np.mean(forces)),
        }
        venv.close()
        eval_env.close()
        print(f" damping={damping:5.1f} door={results[f'damping_{damping}']['mean_door_ang']:.3f} "
              f"success={results[f'damping_{damping}']['success_rate']:.2f} "
              f"K={results[f'damping_{damping}']['mean_K']:.0f}")
    return results


def build_model(algo, venv, seed, tb_dir, device="auto"):
    """Build SB3 model with VIC-suitable hyperparameters."""
    common = dict(verbose=1, seed=seed, tensorboard_log=tb_dir, device=device)
    if algo == "PPO":
        return PPO(
            "MlpPolicy", venv,
            learning_rate=3e-4, n_steps=2048, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.05,  # V3a: entropy annealing 0.05→0.005 via callback
            policy_kwargs=dict(net_arch=[256, 256]),
            **common,
        )
    elif algo == "SAC":
        return SAC(
            "MlpPolicy", venv,
            learning_rate=3e-4, buffer_size=500000, batch_size=256, gamma=0.99, tau=0.005,
            train_freq=1, gradient_steps=1, learning_starts=5000,
            policy_kwargs=dict(net_arch=[256, 256]),
            **common,
        )
    elif algo == "A2C":
        return A2C(
            "MlpPolicy", venv,
            learning_rate=7e-4, n_steps=8,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            **common,
        )
    raise ValueError(f"Unknown algo: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="PPO", choices=list(ALGOS.keys()))
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="", help="Override auto run name")
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--device", default="auto",
                    help="Torch device (auto/cuda/cpu)")
    ap.add_argument("--test-damping", type=float, nargs="*", default=None,
                    help="Damping values for zero-shot eval (e.g., 0.2 0.5 2 5 10)")
    ap.add_argument("--eval-episodes", type=int, default=10,
                    help="Episodes per damping eval")
    ap.add_argument("--curriculum-level", type=int, default=None,
                    help="Curriculum level: 0=10mm/10mm (close), 3=80mm/100mm (default far)")

    # Ablation switches
    ap.add_argument("--no-anti-degeneration", dest="anti_degeneration",
                    action="store_false", default=True,
                    help="Disable anti-degeneration penalties (ablation)")
    args = ap.parse_args()

    # Auto run name
    run_name = args.run_name or f"VIC_{args.algo}_seed{args.seed}"
    run_dir = os.path.join(args.out_root, run_name)
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"=== VIC RUN: {run_name} ===")
    print(f"algo={args.algo} steps={args.steps} seed={args.seed} "
          f"device={args.device}")

    # ── Create VIC environment ──
    env_kwargs = {}
    if not args.anti_degeneration:
        env_kwargs["LAMBDA_K"] = 0.0
        env_kwargs["LAMBDA_F"] = 0.0
        print(" [ablation] anti-degeneration DISABLED")
    if args.curriculum_level is not None:
        env_kwargs["curriculum_level"] = args.curriculum_level

    n_envs = args.n_envs if args.algo in ONPOLICY else 1
    venv = make_vec_env(
        CartesianVICEnv, n_envs=n_envs, seed=args.seed,
        env_kwargs=env_kwargs,
    )

    # ── V1: VecNormalize wrapper (MSG26) ──
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                        clip_reward=10.0, gamma=0.99)
    print(f" [V1] VecNormalize: norm_reward=True, clip_reward=10.0, gamma=0.99")

    # ── Build model ──
    model = build_model(args.algo, venv, args.seed, tb_dir, device=args.device)
    print(f"TensorBoard -> {tb_dir}")

    # ── V2: Optimistic value head bias (MSG30) ──
    with torch.no_grad():
        model.policy.value_net.bias.fill_(5.0)
    print(" [V2] Optimistic value bias: value_net.bias = 5.0")

    # ── V3a: Entropy annealing ONLY (Claude MSG 034) ──
    # Removed unauthorized low-level freeze hooks. Pure entropy scheduling only.
    print(" [V3a] Entropy annealing 0.05→0.005 (no low-level freeze)")

    # ── Train ──
    t0 = time.time()
    anneal_cb = EntropyAnnealingCallback(
        start_ent_coef=0.05, end_ent_coef=0.005,
        total_timesteps=args.steps,
    )
    model.learn(
        total_timesteps=args.steps,
        callback=[VICProgressCallback(log_every=10000), anneal_cb],
        progress_bar=False,
    )
    dt = time.time() - t0
    print(f"Training completed in {dt:.1f}s ({dt/60:.1f} min)")

    # ── Save ──
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)
    print(f"Saved model -> {model_path}")

    pt_path = os.path.join(run_dir, "policy.pt")
    torch.save(model.policy.state_dict(), pt_path)
    print(f"Saved policy.pt -> {pt_path}")

    # Save VecNormalize stats (for eval/reproducibility)
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    venv.save(vn_path)
    print(f"VecNormalize stats saved -> {vn_path}")

    if args.test_damping:
        dampings = sorted(args.test_damping)
        print(f"\n=== Zero-shot damping generalization eval ===")
        results = evaluate_damping_generalization(
        model, dampings, vn_path=vn_path, n_episodes=args.eval_episodes,
        )
        result_path = os.path.join(run_dir, "generalization_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results -> {result_path}")
        print(f"\n{'Damping':>8} {'DoorAng':>8} {'Success':>8} {'K_mean':>8} {'Force':>8}")
        print("-" * 48)
        for k, v in sorted(results.items()):
            print(f"{v['damping']:>8.1f} {v['mean_door_ang']:>8.3f} "
              f"{v['success_rate']:>8.2f} {v['mean_K']:>8.0f} "
              f"{v['mean_contact_force']:>8.1f}")
        venv.close()
        print(f"=== {run_name} done ===")

if __name__ == "__main__":
    main()