#!/usr/bin/env python3
"""
Route B 自主代理 v4 — 全自动：看信箱→理解任务→执行→回信
不需要 Reasonix AI 介入，全自动闭环。
"""

import os, sys, subprocess, re, time, json, shutil
from pathlib import Path
from datetime import datetime

# ── 路径 ──
REPO = Path(__file__).parent.resolve()
MAIN = REPO.parent
MAILBOX = REPO / "AI_CHANNEL.md"
ENV_FILE = MAIN / "xarm7_cartesian_vic" / "cartesian_vic_env.py"
TRAIN_SCRIPT = MAIN / "xarm7_cartesian_vic" / "train_vic.py"
TRAIN_DIR = MAIN / "xarm7_cartesian_vic"
RUNS_DIR = TRAIN_DIR / "runs"
CUDA = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"

# ── 常量 ──
CHECK_INTERVAL = 1800  # 30 分钟（之前 60 分钟太慢）

# 正则：交接棒
RE_HANDOFF = re.compile(r'【交接棒】→\s*(.+?)(?:\n|$)')
RE_MSG_B   = re.compile(r'MSG-B(\d+)')

# ── 工具函数 ──

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def git(*a, to=30):
    try:
        r = subprocess.run(["git","-C",str(REPO)]+list(a), capture_output=True,text=True,timeout=to)
        return r.returncode==0, r.stdout.strip()
    except: return False, ""

def read_box():
    return MAILBOX.read_text(encoding="utf-8") if MAILBOX.exists() else ""

def append_box(t):
    with open(MAILBOX,"a",encoding="utf-8") as f: f.write("\n\n"+t)

def next_bn(t):
    n=13
    for m in RE_MSG_B.findall(t): n=max(n,int(m))
    return n+1

def handoff_to(t):
    for l in reversed(t.splitlines()):
        m=RE_HANDOFF.search(l)
        if m:
            x=m.group(1).strip()
            if any(k in x for k in ["B","显卡机","GPU","看门狗"]): return "B"
            return x
    return None

def run_cmd(cmd, to=7200, cwd=None):
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=to,cwd=cwd or MAIN)
        return r.returncode, r.stdout[-2000:]+r.stderr[-500:]
    except subprocess.TimeoutExpired: return -1,"[timeout]"
    except Exception as e: return -1,str(e)

# ── 技能 1：改 env 参数 ──

def set_param(name, val):
    """改 cartesian_vic_env.py 里的常量"""
    if not ENV_FILE.exists(): return False
    c = ENV_FILE.read_text("utf-8")
    # 匹配 NAME = number 的各种格式
    pat = rf'^({re.escape(name)}\s*=\s*)[0-9.eE+\-]+'
    repl = lambda m: m.group(1) + str(val)
    nc = re.sub(pat, repl, c, count=1, flags=re.MULTILINE)
    if nc == c:
        # 尝试缩进版本
        pat2 = rf'^(\s*{re.escape(name)}\s*=\s*)[0-9.eE+\-]+'
        nc = re.sub(pat2, repl, c, count=1, flags=re.MULTILINE)
    if nc != c:
        ENV_FILE.write_text(nc, "utf-8")
        log(f"SET {name} = {val}")
        return True
    log(f"FAILED to find {name}")
    return False

# ── 技能 2：应用代码补丁 ──

PATCHES = {
    "remove_gating": (
        ("解除 anti-deg gating：高 K 惩罚无条件激活", [
            ("    # High-K penalty + contact force: only when in contact or door moving\n        if in_contact or door_moving:\n            r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2\n            r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2\n        else:\n            r_K_high = 0.0\n            r_force = 0.0",
             "        # High-K penalty: unconditional (gating removed by watchdog)\n        r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2\n        r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2"),
            ("        if in_contact or door_moving:\n            r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2\n            r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2\n        else:\n            r_K_high = 0.0\n            r_force = 0.0",
             "        r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2\n        r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2"),
        ])
    ),
    "raise_low_thresh": (
        ("把 K_LOW_THRESH 从 200 提到 800", [
            ("K_LOW_THRESH = 200.0", "K_LOW_THRESH = 800.0"),
        ])
    ),
    "high_K_penalty_10x": (
        ("把 LAMBDA_K_HIGH 从 0.001 提到 0.01", [
            ("LAMBDA_K_HIGH = 0.001", "LAMBDA_K_HIGH = 0.01"),
        ])
    ),
    "high_K_penalty_100x": (
        ("把 LAMBDA_K_HIGH 提到 0.1", [
            ("LAMBDA_K_HIGH = 0.001", "LAMBDA_K_HIGH = 0.1"),
        ])
    ),
}

def apply_patch(name):
    if name not in PATCHES: return False
    desc, edits = PATCHES[name]
    log(f"Apply patch: {desc}")
    c = ENV_FILE.read_text("utf-8")
    ok = True
    for old, new in edits:
        if old in c:
            c = c.replace(old, new)
            log(f"  replaced: {old[:40]}...")
        else:
            log(f"  NOT FOUND: {old[:40]}...")
            ok = False
    if ok:
        ENV_FILE.write_text(c, "utf-8")
        log(f"Patch {name} applied")
    return ok

# ── 技能 3：训练 ──

def train(steps=300000, curriculum=None, name=None, dampings=None):
    if name is None:
        label = f"curri{curriculum}" if curriculum is not None else "default"
        name = f"VIC_auto_{label}_{steps//1000}k"
    name = name.replace(".","_")
    
    cmd = [CUDA, str(TRAIN_SCRIPT), "--algo","PPO","--steps",str(steps),
           "--n-envs","4","--run-name",name,"--device","cuda"]
    if curriculum is not None and curriculum<999:
        cmd += ["--curriculum-level", str(curriculum)]
    if dampings:
        cmd += ["--test-damping"]+[str(d) for d in dampings]+["--eval-episodes","20"]
    
    log(f"TRAIN {name}")
    t0=time.time()
    code, out = run_cmd(cmd, 7200, TRAIN_DIR)
    dt=time.time()-t0
    
    gp = RUNS_DIR/name/"generalization_results.json"
    if gp.exists():
        mt=datetime.fromtimestamp(gp.stat().st_mtime)
        r=json.loads(gp.read_text())
        ka=sum(v['mean_K'] for v in r.values())/len(r)
        sb=max(v['success_rate'] for v in r.values())
        log(f"DONE {name}: {dt/60:.0f}min K={ka:.0f} SR={sb:.0%} mtime={mt:%H:%M}")
        return {"ok":True,"K":ka,"SR":sb,"mtime":mt,"name":name,"results":r}
    
    log(f"DONE {name}: {dt/60:.0f}min ok={code} (no results)")
    return {"ok":code==0,"name":name}

# ── 技能 4：读结果 ──

def load_results(name):
    gp=RUNS_DIR/name/"generalization_results.json"
    if not gp.exists(): return None
    return json.loads(gp.read_text())

def all_runs():
    if not RUNS_DIR.exists(): return []
    runs=[]
    for d in RUNS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("VIC_"):
            gp=d/"generalization_results.json"
            if gp.exists():
                runs.append((d.stat().st_mtime, d.name, json.loads(gp.read_text())))
    return sorted(runs, key=lambda x: -x[0])

# ── 技能 5：解析任务 ──

def parse_task(text):
    """从信箱文本解析 AI 的指示"""
    tasks = []
    params = {}
    
    # 找参数变更
    for p in ["LAMBDA_K_HIGH","LAMBDA_K_LOW","K_LOW_THRESH","LAMBDA_F","FORCE_THRESHOLD"]:
        m = re.search(rf'{p}\s*[=:]\s*([0-9.]+)', text)
        if m: params[p] = float(m.group(1))
    
    # 找训练参数
    m = re.search(r'(\d+)\s*k', text)
    if m: params["steps"] = int(m.group(1))*1000
    m = re.search(r'curriculum[=_:：\s]*(\d+)', text)
    if m: params["curriculum"] = int(m.group(1))
    
    # 找补丁
    if re.search(r'(?:解除|去除|删|remove|gating|unconditional)', text, re.I):
        tasks.append("remove_gating")
    if re.search(r'(?:K_LOW_THRESH|低K阈值|提到\s*800|threshold.*800)', text):
        tasks.append("raise_low_thresh")
    if re.search(r'(?:LAMBDA_K_HIGH.*0\.0?1|10倍|10x)', text):
        tasks.append("high_K_penalty_10x")
    if re.search(r'(?:LAMBDA_K_HIGH.*0\.?1|100倍|100x)', text):
        tasks.append("high_K_penalty_100x")
    
    # 找动作
    if re.search(r'(?:跑|训|train|run)', text, re.I):
        tasks.append("train")
    if re.search(r'(?:对比|比较|compare|分析|analyze|all.*run)', text, re.I):
        tasks.append("analyze")
    if params:
        tasks.append("set_params")
    
    # 去重
    tasks = list(dict.fromkeys(tasks))
    return tasks, params

# ── 主循环 ──

def cycle(n):
    log(f"=== Cycle {n} ===")
    
    ok,_ = git("pull","origin","session/routeB-cartesian-vic")
    if not ok: log("pull failed"); return
    
    box = read_box()
    ho = handoff_to(box)
    log(f"Handoff -> {ho}")
    
    if ho != "B":
        log("Not my turn, sleep")
        return
    
    log("B's turn! Analyzing AI request...")
    tasks, params = parse_task(box)
    log(f"Tasks: {tasks}, Params: {params}")
    
    results = []
    
    # 1. 改参数
    if "set_params" in tasks:
        for k,v in params.items():
            if k in ["LAMBDA_K_HIGH","LAMBDA_K_LOW","K_LOW_THRESH","LAMBDA_F","FORCE_THRESHOLD"]:
                set_param(k, v)
        results.append(f"Params set: {params}")
    
    # 2. 打补丁
    patches_applied = []
    for t in tasks:
        if t in PATCHES:
            if apply_patch(t):
                patches_applied.append(t)
                results.append(f"Patch applied: {PATCHES[t][0]}")
    
    # 3. 训练
    if "train" in tasks or patches_applied:
        steps = params.get("steps", 300000)
        curriculum = params.get("curriculum", None)
        suffix = "_nogating" if "remove_gating" in tasks else ""
        suffix += f"_khi{params['LAMBDA_K_HIGH']}" if "LAMBDA_K_HIGH" in params else ""
        
        r = train(steps, curriculum, f"VIC_auto{suffix}", list(range(0,11,1))+[0.2,0.5,1,2,5,10])
        # Actually just do standard dampings
        dampings = [0.2,0.5,1.0,2.0,5.0,10.0]
        r = train(steps, curriculum, f"VIC_auto{suffix}", dampings)
        results.append(f"Train: K={r.get('K','?'):.0f} SR={r.get('SR','?'):.0%}")
    
    # 4. 对比分析
    if "analyze" in tasks or not tasks:
        runs = all_runs()
        if len(runs) >= 2:
            table = "\n| Run | K_avg | SR_max |\n|:---|:----:|:-----:|\n"
            for mt, name, rd in runs[:5]:
                ka = sum(v['mean_K'] for v in rd.values())/len(rd)
                sb = max(v['success_rate'] for v in rd.values())
                table += f"| {name} | {ka:.0f} | {sb:.0%} |\n"
            results.append(table)
    
    # ── 写报告 ──
    bn = next_bn(box)
    report = f"## MSG-B{bn} - Agent auto report\n\n"
    for r in results:
        report += r + "\n\n"
    
    # 确定交接棒
    if any("SR=" in r and "0%" in r for r in results):
        report += "【状态】[需AI分析]（训练完成，SR仍低，需调参策略）\n"
        report += "【交接棒】→ 请唤起 AI（SoniXChat）分析结果\n"
    else:
        report += "【状态】[done]（任务完成）\n"
        report += "【交接棒】→ 请唤起 AI（SoniXChat）审核\n"
    
    append_box(report)
    git("add","AI_CHANNEL.md",str(ENV_FILE))
    
    # 检查是否有未提交的改动
    ok, _ = git("diff","--cached","--quiet")
    if not ok:
        git("commit","-m",f"MSG-B{bn}: agent auto {','.join(tasks)}")
        git("push","origin","session/routeB-cartesian-vic")
        log(f"Pushed MSG-B{bn}")
    else:
        log("No changes to commit")

def main():
    log("[AGENT v4] 自主代理启动 (30分钟循环)")
    r=subprocess.run([CUDA,"-c","import torch;print(torch.cuda.get_device_name(0))"],
                     capture_output=True,text=True,timeout=10)
    log(f"GPU: {r.stdout.strip()}")
    
    cycle_n=0
    start=time.time()
    while time.time()-start<24*3600:
        cycle_n+=1
        try:
            cycle(cycle_n)
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback; traceback.print_exc()
        
        rem=(time.time()-start)/3600
        log(f"Run {rem:.1f}h, sleep {CHECK_INTERVAL//60}min\n")
        time.sleep(CHECK_INTERVAL)
    
    log("[AGENT] 24h done")

if __name__=="__main__":
    main()
