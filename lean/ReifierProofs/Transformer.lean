import ReifierProofs.Emit

/-!
# The transformer (`src/reifier/transformer.py`)

```
x = embed[tokens]                    ONE and the token's own code
x = x + attention(norm(x))           head h copies the code of the token h back
x = x + layer(norm(x)), per layer    emit's layers
logits = norm(x) @ embed.T
```

Tokens are a function from positions to token ids; a query at position `p` sees the positions
`p' ≤ p` with `p - p' < CONTEXT`. The softmax is written without Python's shift by the largest
score, which changes nothing in exact arithmetic.
-/

namespace Reifier

open Finset

/-- of dimensions of a query or a key -/
def PAIRS : ℕ := 8
/-- a token attends to itself and the `CONTEXT - 1` before it -/
def CONTEXT : ℕ := 512

noncomputable section

/-- a head scores its token `MARGIN` more than any other in `CONTEXT` -/
def MARGIN : ℝ := 30

/-- `turns = 10000.0 ** -(np.arange(PAIRS) / PAIRS)`: RoPE turns pair `i` by `turn i` radians a
position. -/
def turn (i : ℕ) : ℝ := (10000 : ℝ) ^ (-((i : ℝ) / PAIRS))

/-- `mean(cos(e turns))` -/
def meanCos (e : ℕ) : ℝ := (∑ i ∈ range PAIRS, Real.cos (e * turn i)) / PAIRS

/-- `runner_up(turns, CONTEXT)`: the largest `mean(cos(e turns))` over `0 < e < CONTEXT`. -/
def runnerUp : ℝ := (Icc 1 (CONTEXT - 1)).sup' ⟨1, by simp [CONTEXT]⟩ meanCos

/-- `MARGIN / (PAIRS * (1 - runner_up(turns, CONTEXT)))` -/
def attnScale : ℝ := MARGIN / (PAIRS * (1 - runnerUp))

/-- `lookback_query(h, turns)`: `cos(turns h)` on the first of each pair, `-sin(turns h)` on
the second. -/
def lookback (h d : ℕ) : ℝ :=
  if d < PAIRS then Real.cos (h * turn d)
  else if d < 2 * PAIRS then -Real.sin (h * turn (d - PAIRS)) else 0

/-- Python `rope(x, pos, turns)`: each pair of dimensions `(i, i + PAIRS)` turned by `pos turn i`. -/
def rope (x : ℕ → ℝ) (pos : ℕ) : ℕ → ℝ := fun d =>
  if d < PAIRS then x d * Real.cos (pos * turn d) - x (d + PAIRS) * Real.sin (pos * turn d)
  else if d < 2 * PAIRS then
    x d * Real.cos (pos * turn (d - PAIRS)) + x (d - PAIRS) * Real.sin (pos * turn (d - PAIRS))
  else 0

end

/-- The weights of a program on tokens (Python `Transformer`). -/
structure Transformer where
  /-- the bits of each token (`codes`) -/
  codes : List (ℕ × List Bool)
  /-- `len(embed)`, `max(codes) + 1` -/
  vocab : ℕ
  /-- the bits of a code -/
  bits : ℕ
  /-- the heads: the tokens the program reads -/
  heads : ℕ
  /-- `embed[token, feature]`, tied to the logits -/
  embed : ℕ → ℕ → ℝ
  /-- `q[head, dimension, feature]` -/
  q : ℕ → ℕ → ℕ → ℝ
  /-- `k[dimension, feature]` -/
  k : ℕ → ℕ → ℝ
  /-- `v[bit, feature]` -/
  v : ℕ → ℕ → ℝ
  /-- `o[head, feature, bit]` -/
  o : ℕ → ℕ → ℕ → ℝ
  /-- the feed-forward layers, and the grid that names every feature -/
  net : Network

noncomputable section

/-- Python `transformer(t, codes)`. -/
def transformer (t : Trace) (codes : List (ℕ × List Bool)) : Transformer :=
  let bits := t.outputs.length
  let g := tokenGrid t
  let net := emit t g
  let window := t.inputs.length / bits
  let row0 : Row := g.rows.headD fun _ => none
  { codes, bits, net
    vocab := (codes.map (·.1)).foldl max 0 + 1
    heads := window
    embed := fun τ f =>
      match codes.lookup τ with
      | some c => if f = 0 then net.one
          else if f ≤ bits then (if c.getD (f - 1) false then 1 else -1) else 0
      | none => 0
    q := fun h d f => if h < window ∧ f = 0 then attnScale * lookback h d else 0
    k := fun d f => if d < PAIRS ∧ f = 0 then 1 else 0
    v := fun j f => if j < bits ∧ f = j + 1 then 1 else 0
    o := fun h f j =>
      if h < window ∧ j < bits ∧
          f = featIn g.width row0 (.sig (t.inputs.getD ((window - 1 - h) * bits + j) 0))
      then 1 else 0 }

/-- the positions a query at `p` sees -/
def visible (p : ℕ) : Finset ℕ := (range (p + 1)).filter fun p' => p - p' < CONTEXT

namespace Transformer

variable (M : Transformer)

/-- `embedded`: the grid's row 0 cut to `ONE` and the code. -/
def embedded : Row := fun f => if f ≤ M.bits then (M.net.grid.headD fun _ => none) f else none

/-- `A @ z`, for a matrix `A[row, feature]` -/
def matvec (A : ℕ → ℕ → ℝ) (z : ℕ → ℝ) : ℕ → ℝ := fun d => ∑ f ∈ range M.net.W, A d f * z f

/-- `z = rmsnorm(embed[token], embedded, one)` -/
def zvec (τ : ℕ) : ℕ → ℝ := rmsnorm M.net.W M.embedded M.net.one (M.embed τ)

/-- the key at `p'`: `rope(z @ k.T, p')` -/
def key (toks : ℕ → ℕ) (p' : ℕ) : ℕ → ℝ := rope (M.matvec M.k (M.zvec (toks p'))) p'

/-- the query of head `h` at `p`: `rope(z @ q[h].T, p)` -/
def query (toks : ℕ → ℕ) (h p : ℕ) : ℕ → ℝ := rope (M.matvec (M.q h) (M.zvec (toks p))) p

/-- the score of `p'` for head `h` at `p` -/
def score (toks : ℕ → ℕ) (h p p' : ℕ) : ℝ :=
  ∑ d ∈ range (2 * PAIRS), M.query toks h p d * M.key toks p' d

/-- the softmax over the positions `p` sees -/
def attn (toks : ℕ → ℕ) (h p p' : ℕ) : ℝ :=
  Real.exp (M.score toks h p p') / ∑ p'' ∈ visible p, Real.exp (M.score toks h p p'')

/-- the value at `p'`: `z @ v.T` -/
def value (toks : ℕ → ℕ) (p' : ℕ) : ℕ → ℝ := M.matvec M.v (M.zvec (toks p'))

/-- The stream at `p` after the attention: `x + sum over heads of (a @ values) @ o[h].T`. -/
def afterAttn (toks : ℕ → ℕ) (p : ℕ) : ℕ → ℝ := fun f =>
  M.embed (toks p) f + ∑ h ∈ range M.heads, ∑ j ∈ range M.bits,
    M.o h f j * ∑ p' ∈ visible p, M.attn toks h p p' * M.value toks p' j

/-- The stream at `p` after the last layer (`xs[-1]`), with `gelu` units. -/
def final (toks : ℕ → ℕ) (p : ℕ) : ℕ → ℝ := M.net.final gelu (M.afterAttn toks p)

/-- `rmsnorm(xs[-1], grid[-1], one) @ embed.T`, the logit of token `τ` at `p`. -/
def logit (toks : ℕ → ℕ) (p τ : ℕ) : ℝ :=
  ∑ f ∈ range M.net.W,
    rmsnorm M.net.W (M.net.grid.getLast?.getD fun _ => none) M.net.one (M.final toks p) f
      * M.embed τ f

end Transformer

/-- The bits a program reads at position `p`: for input `k`, bit `k % bits` of the token
`window - 1 - k / bits` back (the oldest token's first). A token with no code reads as zeros. -/
def windowBits (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) : ℕ → Bool :=
  fun k =>
    let b := t.outputs.length
    ((codes.lookup (toks (p + 1 + k / b - t.window))).getD []).getD (k % b) false

end

end Reifier
