# T=7 PB-q1 candidates whose triple has two units with the same A=0 slice gate (mergeable slices)
import pickle, glob, numpy as np, collections
d = pickle.load(open('/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof2/av/units-per-bit/syn/d_p3_T7_1,-1,2,-2.pkl','rb'))
keys, cands = d['keys'], d['cands']
R = np.array(keys, dtype=float); R0 = R[:, :8]
def norm(r):
    s = r.sum()
    return tuple((r / s).round(9)) if s > 0 else None
k0 = [norm(r) for r in R0]
out = set(); n = 0
for fn in sorted(glob.glob('pb_T7_q1_*.pkl')):
    for o in pickle.load(open(fn, 'rb')):
        if o[0] != 'q1':
            continue
        a, b, c = o[1]
        ks = [k0[a], k0[b], k0[c]]
        if any(ks[i] is not None and ks[i] == ks[j] for i in range(3) for j in range(i + 1, 3)):
            out.add((o[1], o[3], o[4]))
print("merge candidates", len(out))
pickle.dump(sorted(out), open('t7merge_cands.pkl', 'wb'))
# also: triples (any) with two equal A=0 slices, regardless of q1 relaxed hit
tr = set()
for h in d['hits']:
    ks = [k0[i] for i in h]
    if any(ks[i] is not None and ks[i] == ks[j] for i in range(3) for j in range(i + 1, 3)):
        tr.add(tuple(h))
print("triples with a repeated slice gate", len(tr))
