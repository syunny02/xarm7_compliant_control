#!/usr/bin/env python3
"""最终受损门评估 - contact-filtered + 4门"""
import sys, numpy as np
sys.path.insert(0,'xarm7_cartesian_vic')
from cartesian_vic_env import CartesianVICEnv
from stable_baselines3 import PPO

MODEL = 'xarm7_cartesian_vic/runs/VIC_300k_v2/model.zip'
N_EP = 20
MAX_STEPS = 300
D = 1.0

DOORS = {
    "baseline": "mujoco_rl/door_real_scene.xml",
    "easy": "xarm7_compliant_control/mujoco_rl/door_broken_hinge_easy.xml",
    "medium": "xarm7_compliant_control/mujoco_rl/door_broken_hinge_medium.xml",
    "hard": "xarm7_compliant_control/mujoco_rl/door_broken_hinge_hard.xml",
}

model = PPO.load(MODEL, device="cuda")

for door_name, xml in DOORS.items():
    env = CartesianVICEnv(xml_path=xml, max_steps=MAX_STEPS, test_damping=D)
    raw, contact_sr, doors, forces = 0, 0, [], []
    for ep in range(N_EP):
        obs,_ = env.reset()
        max_door, max_f = 0, 0
        opened = False
        for _ in range(MAX_STEPS):
            a,_ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(a)
            d = info['door_ang']; f = info['contact_force']
            max_door = max(max_door, d)
            max_f = max(max_f, f)
            if terminated:
                opened = True
                break
            if truncated:
                break
        doors.append(max_door); forces.append(max_f)
        if opened: raw += 1
        if opened and max_f > 0.001: contact_sr += 1
    print(f"{door_name:>8}: raw={raw}/{N_EP}={raw/N_EP:.0%} contact_SR={contact_sr}/{N_EP}={contact_sr/N_EP:.0%} K={info.get('K',0):.0f} F={np.mean(forces):.1f} door={np.mean(doors):.3f}")
    env.close()
