"""rebuild results.jsonl from runs/: one harness line + one audit line per reference set"""
import glob, json, os
out = []
names = sorted({os.path.basename(p)[2:-5] for p in glob.glob("runs/h_*.json")})
for v in names:
    try:
        h = json.loads(open(f"runs/h_{v}.json").readline())
    except Exception:
        continue
    h = {"kind": "harness", "variant": f"sp:{v}", **h}
    out.append(h)
    for tag in ("ref777_w6", "ref_w6"):
        p = f"runs/a_{v}_{tag}.json"
        if os.path.exists(p) and os.path.getsize(p):
            a = json.loads(open(p).readline())
            out.append({"kind": "audit", "ref": tag, **a})
with open("results.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
# summary
for v in names:
    rs = [r for r in out if r.get("variant") == f"sp:{v}"]
    h = [r for r in rs if r["kind"] == "harness"]
    a = [r for r in rs if r["kind"] == "audit"]
    if not h:
        continue
    h = h[0]
    aud = " ".join(f"{r['ref']}:{'ok' if r['ok'] and r['wrong_bits'] == 0 and not r['eager_mismatch'] else 'FAIL'}({r['margin']:.1e})" for r in a)
    print(f"{v:14s} d{h['depth']} dense {h['dense']:>11,} sparse {h['sparse']:>8,} ok {h['ok']} m {h['margin']:.1e} | {aud}")
