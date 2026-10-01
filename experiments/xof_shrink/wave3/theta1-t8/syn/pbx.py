# exact sympy check of PB candidates (chunked): python3 pbx.py T pkl chunk nchunks aff files...
import pickle, sys, glob, time, signal
import pbcheck
T = int(sys.argv[1]); pk = sys.argv[2]; ci, nc = int(sys.argv[3]), int(sys.argv[4]); aff = int(sys.argv[5]); files = sys.argv[6:]
keys, cands = pbcheck.load(T, pk)
seen = set(); cand = []
for fn in sorted(files):
    for o in pickle.load(open(fn, "rb")):
        if o[0] != "q1":
            continue
        key = (o[1], o[3], o[4])
        if key not in seen:
            seen.add(key); cand.append(key)
cand = cand[ci::nc]
print("cands", len(cand), flush=True)
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
nok = 0; nto = 0; t0 = time.time()
for i, (tr, dr, T0) in enumerate(cand):
    gates = [cands[keys[j]] for j in tr]
    signal.alarm(20)
    try:
        r = pbcheck.exact_check(T, gates, dr, T0, bool(aff))
    except TO:
        nto += 1; print("TIMEOUT", gates, dr, T0, flush=True); continue
    finally:
        signal.alarm(0)
    if r[0] == "sols" and r[2]:
        nok += 1
        print("EXACT", gates, dr, T0, [(k, {str(a): str(b) for a, b in s.items()}) for k, s in r[2][:2]], flush=True)
    elif r[0] == "err":
        print("ERR", gates, dr, T0, r[1], flush=True)
    if i % 500 == 0:
        print("checked", i, "ok", nok, "to", nto, round(time.time() - t0), flush=True)
print("DONE ok", nok, "to", nto)
