"""High-precision check of p1d_forms: evaluate every form (its float64 coefficients) in 50-digit
arithmetic at every integer s in [0, n]; also the silu version at k = 32 (the compiler's c*q)."""
import mpmath as mp, sys
sys.path.insert(0, "/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof3/and-of-parities/repo/experiments/xof_shrink/depth_low/lib/python")
import p1d_forms as P
mp.mp.dps = 50
for name, tab in (('FORMS', P.FORMS), ('FORMS_FLAT', P.FORMS_FLAT), ('EXT', {n: P.FORMS_EXT[n] for n in (8, 10, 14, 16, 20, 26, 42)}),
                  ('EXTC', {n: P.FORMS_EXTC[n] for n in (8, 10, 14, 16, 20, 26, 36, 38, 40, 42)}), ('EXTT', {n: P.FORMS_EXTT[n] for n in (10, 14, 20, 26, 42)})):
  for n, (us, c) in sorted(tab.items()):
      e_relu = e_silu = 0
      mind = min(abs(mp.mpf(gw) * s + mp.mpf(gb)) for gw, gb, _, _ in us for s in range(n + 1))
      for s in range(n + 1):
          r = mp.mpf(c) + sum(max(0, mp.mpf(gw) * s + mp.mpf(gb)) * (mp.mpf(vw) * s + mp.mpf(vb)) for gw, gb, vw, vb in us)
          k = 32
          q = mp.mpf(c) + sum((lambda g: g * k / (1 + mp.e ** (-g * k)) / k)(mp.mpf(gw) * s + mp.mpf(gb)) * (mp.mpf(vw) * s + mp.mpf(vb)) for gw, gb, vw, vb in us)
          e_relu = max(e_relu, abs(r - s % 2)); e_silu = max(e_silu, abs(q - s % 2))
      print(f"{name} n={n} units={len(us)} max|err| relu {float(e_relu):.2e} silu(k=32) {float(e_silu):.2e} min|gate| at integers {float(mind):.3f}")
