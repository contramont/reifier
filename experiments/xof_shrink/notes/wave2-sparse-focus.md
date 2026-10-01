# sparse-focus (wave 2): fewer nonzero parameters for the 1-round Keccak XOF MLP

Circuit: `xof(msg, depth=3, k)`, `k = Keccak(log_w=6, n=1, c=448, pad_char="_")`, compiled to
plain `MLP_SwiGLU` (RMSNorm, then `wo(silu(wg n) * wv n)`; no residual, no tying, no
embedding or readout). Builder: `sp.py`, which uses the helpers in `xs.py` and `xs3.py`.
The compiler (`repo/src`) is unchanged. Every number below comes from `xofbench.py` plus
`audit/adv_check.py` on BOTH reference sets (`results.jsonl`: one harness line and two audit
lines per variant). Every listed audit has `ok: true`, 0 wrong bits and no eager mismatch.

## Result

The best sparse point is **91,778 at depth 9, which is 1.19x smaller than 109,101**. At
depth 8 the best is **95,955, 1.14x smaller**. The 2x target is not reachable in this
architecture; the headroom section gives the accounting.

| depth | variant | dense | sparse | vs frontier at the same depth (dense / sparse) | audit margin (ref777 / ref_w6) |
|---|---|---|---|---|---|
| 9 | **`sp:col1b`** | 68,233,410 | **91,778** | new depth; 0.84x the d8 frontier's sparse | 5.0e-5 / 4.3e-5 |
| 9 | `sp:col1b_x2e` | 66,111,490 | 94,658 | | 6.5e-5 / 4.3e-5 |
| 8 | **`sp:base`** | 64,162,035 | **95,955** | +5.0% / **-12.1%** | 1.1e-4 / 1.1e-4 |
| 8 | `sp:base_mp` | 63,204,510 | 97,237 | +3.5% / -10.9% | 2.9e-4 / 2.0e-4 |
| 8 | `sp:x2e` | 62,040,115 | 98,835 | +1.6% / -9.4% | 1.1e-4 / 1.2e-4 |
| 8 | **`sp:x2e_mp`** | **61,082,590** | 100,117 | **same dense / -8.2%: dominates the frontier point** | 2.9e-4 / 2.0e-4 |
| 8 | **`sp:x2e_m3_mp`** | **60,706,707** | 102,929 | **-0.6% / -5.7%: dominates the frontier point** | 2.8e-4 / 4.3e-4 |
| 7 | `sp:lazy2_l3` | 70,156,703 | **114,454** | +10.2% / -10.2% | 2.3e-4 / 1.7e-4 |
| 7 | `sp:lazy2` | 65,316,191 | 118,998 | +2.6% / -6.6% | 3.5e-4 / 3.5e-4 |
| 7 | **`sp:lazy2_m3_mp`** | **63,651,004** | 121,686 | **same dense / -4.5%: dominates the frontier point** | 3.1e-4 / 4.0e-4 |

Frontier (wave 1): d7 `xs3:split_first_lazy4c_m3_mp` 63,651,004 / 127,470; d8
`xs3:split_first_middle_mp` 61,082,590 / 109,101. At the same layer widths, the `_mp` and
`_m3` variants have exactly the frontier's dense, because the new ideas change only which
nonzeros sit in the matrices, not their shapes.
Also measured and audited (dominated): `sp:col1` 9 / 68,233,410 / 92,098 (col1b without the copy trick
below), `sp:col1_x2e` 9 / 66,111,490 / 94,978, `sp:pre` 9 / 71,215,850 / 93,986.
Harness only (ablations, dominated): `sp:nog` (no g-fold) 8 / 64.16M / 102,355,
`sp:e1` (E-fold in round 1) 98,539, `sp:col1_lazy2` 8 / 69.39M / 115,141.

Run (from this directory; `S` is the scratchpad):
`PYTHONPATH=repo/src:repo/experiments/xof_shrink:. $S/venv/bin/python
repo/experiments/xof_shrink/xofbench.py --log-w 6 --depth 3 --variant sp:col1b --widths`,
audit: `./audit.sh col1b` (both reference sets). `eager_check.py sp:<v> 0,1,2,3` compares
the variant's eager function with the reference `xof` at small widths (every variant: 0 bad).

## Where the nonzeros were, and where they are now (per layer; norm + wg + wv + wo)

| layer | frontier d8 (`split_first_middle_mp`) | `sp:base` (d8) | `sp:col1b` (d9) |
|---|---|---|---|
| theta1 | X1 25,305; Y1 7,463 | X1 19,471; Y1 12,631 | col-parity 9,615; D+E 7,967; Y1 10,343 |
| chi1 | 17,706 | 11,306 | 11,306 |
| theta2 | X2 19,751; Y2 8,455 | X2 13,351; Y2 15,175 | 13,351; 15,175 |
| chi2 -> 320 counts | 17,131 | 10,731 | 10,731 |
| theta3 (320 bits) | 8,903 | 8,903 | 8,903 |
| chi3 + digest decode | 4,387 | 4,387 | 4,387 |
| total | 109,101 | 95,955 | 91,778 |

In `sp:col1b` the parameter types are: norm 12,433 (13.5%), wg 23,460, wv 26,635, wo 29,250.
BOS costs only 6 nonzeros per layer (two step units), so "cheaper BOS" has nothing to gain.

## The three ideas, and why each is exact

1. **g-fold: chi reads ONE folded feature.** chi(a, b, c) = a ^ (~b & c) = h(g) with
   g = 2a - b + c in {-1..3} and h(g) = max(0, g)(3 - g)/2. This is exact at every lattice
   point: h(-1) = h(0) = h(3) = 0 and h(1) = h(2) = 1. The layer before chi (the theta Y layer) sums its theta units
   into g in its wo (weights 2, -1, 1, divided by 4 to keep RMSNorm's scale >= 1; read back
   with weight 4). A chi unit then has gate {g} and value {g, 1}, 3 nonzeros instead of
   3 + 4 = 7. An exhaustive search (`chi_search.py`, integer gates in [-4, 4]) shows 7 is the
   minimum for a single chi unit on three bit features. The price is 2 more wo entries per
   theta unit, because each theta bit is a, b and c of three chi bits. Net -2 per state bit
   and chi layer, which is -6.4K over the two full chi layers (`nog` 102,355 -> `base`
   95,955). Negations and iota become a constant in g, which goes into the chi unit's
   biases. It is not worth it for chi3, where theta3 has 6 units per bit: +2.1K there
   against -0.9K saved.
2. **D-sep: the theta X layer emits D = parity(P) of each column pair as ONE feature.**
   Before, D's 5 units were added into the 5 E = a + D features (25 wo entries per pair).
   Now they are added into D only (5 entries), and Y computes t = a xor D with one unit
   max(0, a + D)(2 - a - D) (5 nonzeros instead of 3 for [E == 1] on E). Per pair of
   5 bits: -20 wo, +10 gate/value, +1 norm. Measured: -2.6K in round 1 (`e1` -> `base`) and
   -2.9K in round 2 (`x2e` -> `base`). The trade flips when D is a single unit; see 3.
3. **col1: round 1 from column parities (one more layer).** A message bit used to be read by
   the gates of about 4 glu_xor units of each of the two column pairs that contain it (D
   from raw bits, 7 live bits per pair, 42 nonzeros per pair). Now layer A computes each
   column's parity (3-4 live bits, 2 units), so each bit is read by one parity only, and
   copies the message. Layer B computes D = c(x-1, z) xor c(x+1, z+1) with ONE unit per
   pair. That one unit is cheap to add into the five E = a + D (5 wo), so Y1 reads one
   feature. Round 1 goes from 32.1K (X1 + Y1) to 27.9K in three layers. `col1b` also
   reads the E that have no message bit (E = D in {0, 1}, 320 of them) with a copy unit
   (2 nonzeros instead of 3): -320.
   Exact: every unit is glu_xor or copy on exact integers, with knots on integers.
Also used: glu_xor instead of min-parity wherever sparse matters. Min-parity values read
all n inputs, so `_mp` saves units (dense) and adds nonzeros. Digests are carried as pairs:
3-bit packs (`_m3`) cut dense and add sparse.
Dense-matched points: `x2e_mp` and `x2e_m3_mp` keep the frontier's X2 layout (E-fold, so
the frontier's widths) and add g-fold and D-sep to round 1 only. `lazy2*` is the frontier's
depth-7 layout (lazy chi2) with g-fold and D-sep in round 1. `lazy2_l3` uses the 3-unit
lazy chi (count range 22), which is 4.5K sparser than lazy4c at depth 7.

All audit margins are at most 4.3e-4, well below the 0.02 limit. The sparse-best
layouts (`base`, `col1b`) are at most 1.1e-4. They also pass the harness with 128 random
messages (margins 6.0e-5 and 4.2e-5, `runs/final/h128_*.json`), and the final `sp.py`
reproduces their recorded metrics (`runs/final/`).

## What did not work, and why (measured unless marked "est.")

- E-fold in round 1 (`e1`): +2.6K. D has 4 units there, so adding it into the E features
  costs more wo than Y saves.
- E-fold in round 2 (`x2e`): +2.9K sparse, but -2.1M dense. It is the dense-side option.
- A linear pre-layer for round 1 (`pre`: pass units compute each pair sum P from the
  message bits, then D = parity(P)): 93,986 at depth 9, worse than col1's 92,098.
- A lazy round 2 at depth 8 (`col1_lazy2`): 115K. The wide counts of the lazy chi cost
  more in theta3 than the Y layer they remove.
- (est.) Round 2 from column parities, as in col1 (4 layers): +1.8K. The extra layer of
  1600 copies (4.8K + 1.9K norm) eats the saving. Round 1 gains only because reading raw
  message bits twice was expensive.
- (est.) Column parities in the last round (split theta3, E3 = own + c1 + c2): +0.6K and
  one more layer. D3 is used once there, so nothing is shared.
- (est.) Round 1 in two layers from column parities, with a 3-input xor (2 units) in Y1:
  +2K against `base`.
- (est.) Emitting column sums instead of pair sums from chi: -1600 wo, +1920 reads in the
  D units (net +0.3K).
- (est.) xor as a product unit plus pass terms, s (1 - 2D) plus a shared D: 47 against 40
  nonzeros per pair. Bipolar encodings need 2 units per xor.
- (est.) Packing digests 3-4 bits per feature: decoders cost 5+ units per bit. Pairs are
  the sparse optimum: carry 4 per feature per layer, decode 9 per pair.
- (est.) The AND of two adjacent chi bits to cut theta3's count range 11 -> 10: the AND
  unit reads 4 theta bits, which g-fold no longer provides, so it costs more than the one
  saved unit.

## Headroom (my view, with lower-bound reasoning)

The best layout for sparse is three layers per full round: X (copy a, D units), Y (xor,
g-fold), chi (g). One layer fewer means one-layer theta (6+ units per bit) or lazy chi
(wide counts): both measured worse. One layer more adds a layer of 1600 copies. Per full
round and state bit, the accounting is:

- norms: 3 layers x about 1.1 features per bit, about 3.3;
- chi: one unit (3 nonzeros; 7 is the minimum without the fold) plus fan-out 3 (own bit and
  two pair sums);
- X: copy (2 + 1) plus D (15 nonzeros per pair of 5 bits plus 1 wo). Parity of a count in
  [0, 10] needs 5 units with integer knots, and 15 nonzeros is the minimum for them
  (`par_search.py`: exhaustive over every set of 5 left- or right-opening integer knots
  and every value sparsity pattern). Min-parity does not help at 10 and adds value
  nonzeros;
- Y: xor (5; it is the minimum for one unit on two bit features) plus fan-out 3.

That is about 24.5 per bit (measured: round 2 is 39.9K / 1600 = 24.9). The fan-outs
(3 + 3), the copies and the norms (about 9.3 together) are forced by the no-residual
architecture and by Keccak's diffusion: each bit feeds its own position and two column
pairs, and each theta bit feeds three chi bits. Units are at their single-unit minimum
nonzero counts. So a full round cannot go much below about 22 per bit in this family.
Round 1 is about 28K (1464 theta outputs) and the last round about 24K (1600 chi2 bits into
320 counts, 320 theta3, 224 chi3, digest decoding about 4K). Summing gives a floor of
about 85-88K, against 91.8K now: **about 5-7% headroom on sparse** without a new idea at
the representation level. 2x (about 55K) would need about 12 per bit per round. Removing
the copies and the norm of carried features would need a residual stream, and removing the
fan-outs would need shared or tied weights. Both are outside the fixed architecture
(wave 1's gated-residual, tied variant reached 56.7K).

Dense: the sparse-best variants have 3-7M more dense than the frontier, because D-sep
widens X2's output by 320. The dense-matched variants (`x2e_mp`, `lazy2_m3_mp`) show the
ideas cost no dense where the layout is kept. The dense floor analysis of wave 1 (about
50M at depth 8) is unchanged by this work.

Composes with: any cheaper theta or chi unit (the fold is independent of the unit form);
the depth-4/5 layouts only through their round-1 and chi1 layers. theta1-direct is
unaffected: g-fold would cost 2 wo per theta1 unit, 4-5 of them per bit. Also cheaper
decoders for the digests.

## Files

- `sp.py`: the builder. Variants `sp:base`, `sp:col1b`, `sp:x2e_mp`, `sp:lazy2_l3`, ...
  (options: gfold1/2, col1, pre, e1, lazy2 (True = lazy4c, "l3" = 3-unit lazy chi), x2e,
  mp, m3, ebit; D-sep is the default X layer).
- `results.jsonl`: harness plus both audits per variant (`collect.py` rebuilds it from
  `runs/`); `runs/`: raw outputs and per-layer breakdowns.
- `layerstats.py` (per-layer nonzeros), `eager_check.py` (eager vs reference xof; the main
  variants also pass at log_w 4 and 5, `runs/eager_w45.txt`), `chi_search.py` (one-unit
  chi minimum), `par_search.py` (5-unit parity minimum), `audit.sh`, `collect.py`,
  `mkpatch.sh`.
- Steepness and bugs: no compiler change was needed. Everything runs at the base's
  c = 4, q = 8, and the sizes do not depend on q. The only units with knots between
  integers (min-parity, `_mp`) read raw message bits.
- `patch.diff`: `git -C repo diff` with the new files added under
  `experiments/xof_shrink/` (`sp.py`, `sparse_focus/`). The compiler is unchanged, so the
  diff has only new files.
