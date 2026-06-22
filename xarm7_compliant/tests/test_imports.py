#!/usr/bin/env python3
"""验证完整流水线可导入并初始化."""
import sys
sys.path.insert(0, ".")

import numpy as np

print("[TEST] 导入核心模块...")
from xarm7_compliant.controller import CompliantController, CompliantControllerConfig
from xarm7_compliant.core.admittance import AdmittanceFilter, VariableAdmittance
from xarm7_compliant.core.policy import LoadedPolicy
from xarm7_compliant.core.wrench import WrenchMapper
from xarm7_compliant.sim_to_real import SimToRealWrapper
print("[PASS] 全部导入成功")

print("[TEST] 初始化 CompliantController...")
ctrl = CompliantController(CompliantControllerConfig())
ctrl.reset(np.zeros(7))
print("[PASS] Controller init + reset OK")

print("[TEST] 导纳滤波器...")
adm = AdmittanceFilter(7)
adm.reset(np.zeros(7))
q = adm.update(np.ones(7), np.zeros(7))
assert np.allclose(q, np.ones(7), atol=1e-6)
print("[PASS] AdmittanceFilter update OK")

print("[TEST] SimToRealWrapper...")
st = SimToRealWrapper()
st.reset()
print("[PASS] SimToRealWrapper OK")

print("\n=== ALL PIPELINE TESTS PASSED ===")
