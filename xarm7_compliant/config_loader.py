"""
config_loader.py — YAML 配置加载器
====================================
将 YAML 配置文件递归填充到 CompliantControllerConfig 及其子 dataclass.

用法:
    from xarm7_compliant.config_loader import load_config
    cfg = load_config("config/door_opening.yaml")
    ctrl = CompliantController(cfg)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
import yaml

from .controller import CompliantControllerConfig, SafetyLimits
from .core.admittance import AdmittanceConfig
from .core.wrench import WrenchConfig

T = TypeVar("T")


def _dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """递归将字典填充到 dataclass (仅匹配已知字段)."""
    import inspect
    from dataclasses import fields

    valid_keys = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid_keys:
            continue
        field_type = cls.__annotations__.get(key)
        if field_type is None:
            kwargs[key] = value
        elif hasattr(field_type, "__dataclass_fields__"):
            # 嵌套 dataclass
            kwargs[key] = _dataclass_from_dict(field_type, value)
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
            # Dict[str, X] — 透传
            kwargs[key] = value
        elif isinstance(value, list):
            kwargs[key] = tuple(value) if field_type and "tuple" in str(field_type) else value
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str) -> CompliantControllerConfig:
    """从 YAML 文件加载控制器配置.

    Args:
        path: YAML 文件路径 (相对于项目根或绝对路径)

    Returns:
        CompliantControllerConfig: 完整填充的配置对象
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return _parse_config(raw)


def load_config_from_string(yaml_str: str) -> CompliantControllerConfig:
    """从 YAML 字符串加载配置 (用于程序内构造)."""
    raw: Dict[str, Any] = yaml.safe_load(yaml_str)
    return _parse_config(raw)


def _parse_config(raw: Dict[str, Any]) -> CompliantControllerConfig:
    """解析 YAML 顶层结构到 config dataclass."""

    # ── 控制器参数 ──
    ctrl_raw = raw.get("controller", {})
    cfg = CompliantControllerConfig(
        ndof=ctrl_raw.get("ndof", 7),
        device=ctrl_raw.get("device", "cpu"),
        control_dt=1.0 / ctrl_raw.get("control_freq", 100.0),
        use_mujoco=ctrl_raw.get("use_mujoco", True),
        scene_xml=ctrl_raw.get("scene_xml", ""),
        tcp_body_name=ctrl_raw.get("tcp_body_name", "link_tcp"),
        policy_path=ctrl_raw.get("policy_path", ""),
        cartesian_vic=ctrl_raw.get("cartesian_vic", False),
        stiffness_from_policy=ctrl_raw.get("stiffness_from_policy", True),
    )

    # ── 导纳参数 ──
    adm_raw = raw.get("admittance", {})
    cfg.admittance = _dataclass_from_dict(AdmittanceConfig, adm_raw)

    # ── 变量导纳 (覆盖固定参数) ──
    var_raw = raw.get("variable_admittance", {})
    cfg.variable_admittance = var_raw.get("enabled", False)
    cfg.task_phase = var_raw.get("initial_phase", 0)

    # ── 力/力矩传感器 ──
    wrench_raw = raw.get("wrench", {})
    cfg.wrench = _dataclass_from_dict(WrenchConfig, wrench_raw)

    # ── 安全限制 ──
    safety_raw = raw.get("safety", {})
    cfg.safety = _dataclass_from_dict(SafetyLimits, safety_raw)

    return cfg
