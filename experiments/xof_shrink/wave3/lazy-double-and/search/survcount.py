import sys, time
sys.argv = ["dand5.py", "3"]
import dand5 as E
for f in E.D.FLIPS:
    E.czf(f, 5)
t0 = time.time()
for i in E.G1[:: len(E.G1) // 12][:12]:
    print(E.GATES[i][:2], [len(E.survivors(i, f)) for f in E.D.FLIPS], f"{time.time() - t0:.1f}s", flush=True)
