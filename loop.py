#!/usr/bin/env python3
"""循环：每5分钟查信箱，B的回合就干活，否则继续等。跑24小时自动停。"""
import subprocess, time, json, re, os
from pathlib import Path
from datetime import datetime

D = Path(__file__).parent.resolve()
M = D.parent
BOX = D / "AI_CHANNEL.md"
ENV = M / "xarm7_cartesian_vic" / "cartesian_vic_env.py"
TRAIN = M / "xarm7_cartesian_vic" / "train_vic.py"
RUNS = M / "xarm7_cartesian_vic" / "runs"
CUDA = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def git(*a):
    r = subprocess.run(["git","-C",str(D)]+list(a), capture_output=True,text=True,timeout=30)
    return r.returncode==0, r.stdout.strip()

def update_check():
    ok, _ = git("pull","origin","session/routeB-cartesian-vic","--rebase","--autostash")
    if not ok:
        git("stash","pop")
    t = BOX.read_text("utf-8") if BOX.exists() else ""
    for l in reversed(t.splitlines()):
        m = re.search(r'【交接棒】→\s*([^\n]+)', l)
        if m:
            x = m.group(1)
            if any(k in x for k in ["B","显卡机","GPU"]): return True
            return False
    return False

def train_and_report():
    """Run training and report results"""
    name="VIC_auto_loop"
    cmd=[CUDA,str(TRAIN),"--algo","PPO","--steps","300000","--n-envs","4",
         "--run-name",name,"--device","cuda",
         "--test-damping","0.2","0.5","1.0","2.0","5.0","10.0","--eval-episodes","20"]
    log("TRAIN start")
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=7200)
    gp=RUNS/name/"generalization_results.json"
    result="Training done"
    if gp.exists():
        d=json.loads(gp.read_text())
        ka=sum(v['mean_K'] for v in d.values())/len(d)
        sb=max(v['success_rate'] for v in d.values())
        result=f"K_avg={ka:.0f} SR_max={sb:.0%}"
        log(result)
    return result

t0=time.time()
cycle=0
while time.time()-t0<24*3600:
    cycle+=1
    t=datetime.now()
    is_b = update_check()
    if is_b:
        log(f"C{cycle} B's turn! Working...")
        res = train_and_report()
        git("add","AI_CHANNEL.md",str(ENV))
        git("commit","-m",f"auto C{cycle}: {res}")
        git("push","origin","session/routeB-cartesian-vic")
    else:
        log(f"C{cycle} not B (sleep 5min)")
    time.sleep(300)

log("24h done")
