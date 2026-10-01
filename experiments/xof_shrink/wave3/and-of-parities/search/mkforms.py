"""Pick one form per n from logs/best_*.log (min knot distance >= DMIN, then the smallest
cancelling terms) and write depth_low/lib/python/p1d_forms.py (+ an exactness check)."""
import json, glob, sys
DMIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
COUNT_N = (10, 20, 26)
OUT = "/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof3/and-of-parities/repo/experiments/xof_shrink/depth_low/lib/python/p1d_forms.py"
best = {}
for f in glob.glob("logs/best*.log"):
    for line in open(f):
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d["min_knot_dist"] < DMIN or d["max_term"] > 1000:
            continue  # far-from-integer knots only; no huge cancelling terms (float32)
        n = d["n"]
        if n in COUNT_N and "max_slope" not in d:
            continue
        # count parities (noisy inputs): flattest at the integers first; raw parities: smallest terms
        key = (d["K"], d["max_term"] * d["max_slope"]) if n in COUNT_N else (d["K"], d["max_term"])
        if n not in best or key < best[n][0]:
            best[n] = (key, d)
lines = ['"""and-of-parities (wave 3): parity of s = the count of n raw bits, s in [0, n], as',
         'c + sum_k max(0, gw s + gb) (vw s + vb) with knots at irrational points between integers',
         '(found by variable projection over the knots, polished to float64; see search/). Gates are',
         'scaled so that every integer is >= 1 away from each knot in gate units. Exact in float64',
         'to ~1e-13 on every integer s in [0, n] (check() below)."""', "", "FORMS = {"]
for n in sorted(best):
    d = best[n][1]
    lines.append(f"    {n}: ({[tuple(u) for u in d['units']]!r}, {d['const']!r}),  # K={d['K']} knots {[round(k, 4) for k in d['knots']]} max term {d['max_term']:.1f}" + (f" max slope {d['max_slope']:.2f}" if 'max_slope' in d else ""))
lines += ["}", "", "", "def check(n):",
          "    us, c = FORMS[n]",
          "    return max(abs(c + sum(max(0.0, gw * s + gb) * (vw * s + vb) for gw, gb, vw, vb in us) - s % 2) for s in range(n + 1))",
          "", "", 'if __name__ == "__main__":', "    for n in FORMS:", "        print(n, len(FORMS[n][0]), check(n))", ""]
open(OUT, "w").write("\n".join(lines))
print(open(OUT).read())
