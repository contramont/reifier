# exact check of T=7 PB-q1 candidates whose triple has a unit that vanishes at A = 0
# (pool = 2 slices + 1 hinge: 3 per live bit + 3 per column)
import pickle, sys, glob, time, signal, numpy as np
import pbcheck
T = 7; pk = sys.argv[1]; ci, nc = int(sys.argv[2]), int(sys.argv[3])
keys, cands = pbcheck.load(T, pk)
R = np.array(keys); zero0 = set(np.where(R[:, :8].sum(1) == 0)[0].tolist())
seen = set(); cand = []
for fn in sorted(glob.glob("pb_T7_q1_*.pkl")):
    for o in pickle.load(open(fn, "rb")):
        if o[0] == "q1" and any(i in zero0 for i in o[1]):
            key = (o[1], o[3], o[4])
            if key not in seen:
                seen.add(key); cand.append(key)
cand = cand[ci::nc]
print("cands", len(cand), flush=True)
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
nok = 0; t0 = time.time()
for i, (tr, dr, T0) in enumerate(cand):
    gates = [cands[keys[j]] for j in tr]
    signal.alarm(30)
    try:
        r = pbcheck.exact_check(T, gates, dr, T0, False)
    except TO:
        print("TIMEOUT", gates, dr, T0, flush=True); continue
    finally:
        signal.alarm(0)
    if r[0] == "sols" and r[2]:
        nok += 1
        print("EXACT", gates, dr, T0, [(k, {str(a): str(b) for a, b in s.items()}) for k, s in r[2][:2]], flush=True)
print("DONE ok", nok, round(time.time() - t0))
