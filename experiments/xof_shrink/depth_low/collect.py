"""results.jsonl from runs/: harness (h_<tag>.json) and both audits (adv777_, advc1_)"""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
tags = sys.argv[1:]
out = []
for tag in tags:
    rec = {"tag": tag}
    for src, fn in (("harness", f"h_{tag}.json"), ("audit_ref777_w6", f"adv777_{tag}.json"),
                    ("audit_combined1_ref_w6", f"advc1_{tag}.json")):
        p = os.path.join(D, "runs", fn)
        if os.path.exists(p) and os.path.getsize(p):
            r = json.loads(open(p).read().strip().splitlines()[-1])
            r["source"], r["tag"] = src, tag
            out.append(r)
with open(os.path.join(D, "results.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
for r in out:
    print(r["tag"], r["source"], r.get("depth"), r.get("dense"), r.get("sparse"), "margin", r.get("margin"),
          "wrong", r.get("wrong_bits"), "ok", r.get("ok"), "eager_mismatch", r.get("eager_mismatch"))
