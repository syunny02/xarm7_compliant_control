#!/usr/bin/env python3
"""5분 타이머 - Reasonix 세션用"""
import time, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
while True:
    time.sleep(300)
    n += 1
    print(f"[tick {n}] 5min")
