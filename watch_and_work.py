#!/usr/bin/env python3
"""
Route B Agent v5 — 30分钟循环：看信箱→行动→回信
"""
import os, sys, subprocess, re, time, json
from pathlib import Path
from datetime import datetime

D = Path(__file__).parent.resolve()
M = D.parent  # main repo
BOX = D / "AI_CHANNEL.md"
ENV = M / "xarm7_cartesian_vic" / "cartesian_vic_env.py"
TRAIN = M / "xarm7_cartesian_vic" / "train_vic.py"
RUNS = M / "xarm7_cartesian_vic" / "runs"
CUDA = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"

RE_B = re.compile(r'MSG-B(\d+)')
RE_HO = re.compile(r'【交接棒】→\s*([^\n]+)')

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def git(*a):
    try:
        r = subprocess.run(["git","-C",str(D)]+list(a),capture_output=True,text=True,timeout=30)
        return r.returncode==0, r.stdout.strip()
    except: return False, ""

def read_box():
    return BOX.read_text("utf-8") if BOX.exists() else ""

def append_box(t):
    with open(BOX,"a",encoding="utf-8") as f: f.write("\n\n"+t)

def next_bn(t):
    n=13
    for m in RE_B.findall(t): n=max(n,int(m))
    return n+1

def is_b_turn(t):
    for l in reversed(t.splitlines()):
        m=RE_HO.search(l)
        if m:
            x=m.group(1)
            if any(k in x for k in ["B","显卡机","GPU"]): return True
            return False
    return False

def run(cmd,to=7200):
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=to)
        return r.returncode, r.stdout[-2000:]+r.stderr[-500:]
    except Exception as e: return -1,str(e)

def set_env(name,val):
    c=ENV.read_text("utf-8")
    nc=re.sub(rf'^{re.escape(name)}\s*=\s*[0-9.eE+\-]+',f"{name} = {val}",c,flags=re.MULTILINE)
    if nc==c:
        nc=re.sub(rf'(\s{re.escape(name)}\s*=\s*)[0-9.eE+\-]+',f"\\1{val}",c)
    if nc!=c:
        ENV.write_text(nc,"utf-8")
        log(f"SET {name}={val}")
        return True
    log(f"FAIL set {name}")
    return False

def do_train(steps=300000,curriculum=None,name=None):
    if name is None:
        tag=f"curri{curriculum}" if curriculum is not None else "def"
        name=f"VIC_auto_{tag}_{steps//1000}k"
    name=name.replace(".","_")
    cmd=[CUDA,str(TRAIN),"--algo","PPO","--steps",str(steps),"--n-envs","4",
         "--run-name",name,"--device","cuda",
         "--test-damping","0.2","0.5","1.0","2.0","5.0","10.0","--eval-episodes","20"]
    if curriculum is not None and curriculum<999:
        cmd+=["--curriculum-level",str(curriculum)]
    log(f"TRAIN {name}")
    t0=time.time()
    code,out=run(cmd)
    dt=time.time()-t0
    gp=RUNS/name/"generalization_results.json"
    if gp.exists():
        r=json.loads(gp.read_text())
        ka=sum(v['mean_K'] for v in r.values())/len(r)
        sb=max(v['success_rate'] for v in r.values())
        db=max(v['mean_door_ang'] for v in r.values())
        mt=datetime.fromtimestamp(gp.stat().st_mtime)
        log(f"DONE {name}: {dt/60:.0f}m K={ka:.0f} SR={sb:.0%} door={db:.3f} @{mt:%H:%M}")
        tbl="|Damping|Door|SR|K|Force|\n|:-:|:-:|:-:|:-:|:-:|\n"
        for k,v in sorted(r.items(),key=lambda x:x[1]['damping']):
            tbl+=f"|{v['damping']}|{v['mean_door_ang']:.3f}|{v['success_rate']:.0%}|{v['mean_K']:.0f}|{v['mean_contact_force']:.1f}|\n"
        return {"ok":True,"K":ka,"SR":sb,"door":db,"table":tbl,"name":name}
    log(f"DONE {name}: {dt/60:.0f}m code={code}")
    return {"ok":code==0,"name":name}

def cycle(n):
    log(f"=== Cycle {n} ===")
    ok,_=git("pull","origin","session/routeB-cartesian-vic")
    if not ok: log("pull fail"); return
    
    box=read_box()
    if not is_b_turn(box):
        log("Not B's turn"); return
    
    log("B's turn! Working...")
    reports=[]
    
    # 读取 AI 最新消息中的参数变更指示
    # 只取最后一条 MSG-A* 的内容
    msgs=re.split(r'\n## MSG-', box)
    last_ai_msg=""
    for m in reversed(msgs):
        if m.startswith('A'):
            last_ai_msg="MSG-"+m
            break
    
    # 解析参数
    params={}
    for p in ["LAMBDA_K_HIGH","LAMBDA_K_LOW","K_LOW_THRESH","LAMBDA_F","FORCE_THRESHOLD"]:
        m=re.search(rf'{p}\s*[=:]\s*([0-9.]+)', last_ai_msg)
        if m: params[p]=float(m.group(1))
    
    # 解析训练参数
    steps=300000
    m=re.search(r'(\d+)\s*k', last_ai_msg)
    if m: steps=int(m.group(1))*1000
    
    curriculum=None
    m=re.search(r'curricu[^ ]*\s*[=:]\s*(\d+)', last_ai_msg, re.I)
    if m: curriculum=int(m.group(1))
    
    # 执行参数变更
    if params:
        for k,v in params.items():
            if set_env(k,v):
                reports.append(f"Set {k} = {v}")
    
    # 打补丁 - 检测 "去除" "gating" 等关键词
    if re.search(r'(?:去[除掉]|remove|gating|无条)', last_ai_msg, re.I):
        c=ENV.read_text("utf-8")
        old="if in_contact or door_moving:\n            r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2\n            r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2\n        else:\n            r_K_high = 0.0\n            r_force = 0.0"
        new="r_K_high = -self.LAMBDA_K_HIGH * max(0.0, K - self.K_TARGET) ** 2\n        r_force = -self.LAMBDA_F * max(0.0, force_norm - self.FORCE_THRESHOLD) ** 2"
        if old in c:
            c=c.replace(old,new)
            ENV.write_text(c,"utf-8")
            reports.append("Removed anti-deg gating (ungated)")
            log("Gating removed")
        else:
            log("Gating pattern not found (maybe already removed)")
    
    # 涨 K_LOW_THRESH
    if re.search(r'(?:K_LOW_THRESH|低K[阈阈])', last_ai_msg) and re.search(r'(?:80?0|提[高升])', last_ai_msg):
        set_env("K_LOW_THRESH",800.0)
        reports.append("K_LOW_THRESH -> 800")
    
    # 训练
    tag="auto"
    if params.get("LAMBDA_K_HIGH"): tag+=f"_khi{params['LAMBDA_K_HIGH']}"
    if curriculum is not None: tag+=f"_c{curriculum}"
    
    r=do_train(steps,curriculum,f"VIC_{tag}")
    if r.get("table"):
        reports.append(r["table"])
        reports.append(f"K_avg={r['K']:.0f} SR_max={r['SR']:.0%}")
    
    # 写报告
    bn=next_bn(box)
    rep=f"## MSG-B{bn} - Agent report\n\n"+"\n".join(reports)
    rep+="\n\n【状态】[done]\n"
    
    if r.get("SR",0)<0.2:
        rep+="【交接棒】→ 请唤起 AI（SoniXChat）分析：SR偏低需调整\n"
    else:
        rep+="【交接棒】→ 请唤起 AI（SoniXChat）审核结果\n"
    
    append_box(rep)
    git("add","AI_CHANNEL.md",str(ENV))
    ok,_=git("diff","--cached","--quiet")
    if not ok:
        git("commit","-m",f"MSG-B{bn}: agent done")
        git("push","origin","session/routeB-cartesian-vic")
        log(f"Pushed MSG-B{bn}")
    else:
        log("No changes")

def main():
    r=subprocess.run([CUDA,"-c","import torch;print(torch.cuda.get_device_name(0)or'none')"],
                     capture_output=True,text=True,timeout=10)
    log(f"AGENT v5 | GPU: {r.stdout.strip()}")
    n=0
    t0=time.time()
    while time.time()-t0<24*3600:
        n+=1
        try: cycle(n)
        except Exception as e:
            log(f"ERR: {e}")
        log(f"Sleep 30min\n")
        time.sleep(1800)
    log("Done 24h")

if __name__=="__main__":
    main()
