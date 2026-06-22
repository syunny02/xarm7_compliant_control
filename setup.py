#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="xarm7_compliant",
    version="1.0.0",
    description="xArm7 Compliant Control — Sim-to-Real door opening",
    author="Davied (Reinforcement Learning)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "torch>=1.10",
        "pyyaml>=6.0",
    ],
    extras_require={
        "sim": ["mujoco>=3.0", "matplotlib>=3.5"],
        "real": ["rclpy>=3.0", "sensor_msgs", "geometry_msgs"],
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "compliant_demo_sim=xarm7_compliant.examples.demo_mujoco:run_demo",
        ],
    },
)
