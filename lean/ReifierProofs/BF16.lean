import ReifierProofs.Transformer

/-!
# The forward pass in bf16

bf16 numbers are `m * 2^e` with `|m| < 2^8` and `-133 ≤ e ≤ 120`: 8 significant bits, the
exponents of float32, subnormals down to `2^-133`, and the largest finite `255 * 2^120`.

A `Rounding` is any function that rounds each real of finite range to a nearest bf16 number,
whatever it does with ties: IEEE round-to-nearest-even is one (`round_to_bf16` in `gemma.py`).

The model of the forward pass: every weight is rounded to bf16, and so is the result of every
operation of `transformer.py`'s `run`, each computed exactly and rounded once (as a matmul
accumulates in float32 and is stored in bf16): the embedding, each norm, each matmul, RoPE, the
scores, the softmax, `gelu`, the product of gate and value, and each addition to the stream.
-/

namespace Reifier

open Finset

noncomputable section

/-- The largest finite bf16 number, `(2^8 - 1) * 2^120`. -/
def maxBF16 : ℝ := 255 * 2 ^ 120

/-- `x` is a (finite) bf16 number. -/
def IsBF16 (x : ℝ) : Prop :=
  ∃ m e : ℤ, x = m * (2 : ℝ) ^ e ∧ |m| < 256 ∧ -133 ≤ e ∧ e ≤ 120

/-- Rounding to a nearest bf16 number, on reals of finite range. -/
structure Rounding where
  fl : ℝ → ℝ
  fl_mem : ∀ x, |x| ≤ maxBF16 → IsBF16 (fl x)
  fl_nearest : ∀ x r, |x| ≤ maxBF16 → IsBF16 r → |fl x - x| ≤ |r - x|

namespace BF16

variable (R : Rounding)

/-- `one`, as a bf16 weight. -/
def one (N : Network) : ℝ := R.fl N.one

/-- The norm before a layer, in one rounding: Python's `rmsnorm`, where `one` is the bf16 one
that the stream holds. -/
def norm (W : ℕ) (one : ℝ) (row : Row) (x : ℕ → ℝ) : ℕ → ℝ :=
  fun f => R.fl (rmsnorm W row one x f)

/-- A layer on the normed stream `z`, at feature `f`: `fl(w_out @ fl(fl(gelu(fl(w_gate @ z))) *
fl(w_val @ z)))`, every weight rounded. -/
def layerApply (W : ℕ) (us : List Hidden) (z : ℕ → ℝ) (f : ℕ) : ℝ :=
  R.fl (us.map fun u =>
      let G := R.fl (∑ g ∈ range W, R.fl (u.gate g) * z g)
      let V := R.fl (∑ g ∈ range W, R.fl (u.val g) * z g)
      R.fl (u.out f) * R.fl (R.fl (gelu G) * V)).sum

/-- `fl(x + layer(norm(x)))` -/
def netStep (W : ℕ) (one : ℝ) (before : Row) (us : List Hidden) (x : ℕ → ℝ) : ℕ → ℝ :=
  fun f => R.fl (x f + layerApply R W us (norm R W one before x) f)

/-- The stream after the last layer. -/
def final (N : Network) (x : ℕ → ℝ) : ℕ → ℝ :=
  (N.grid.zip N.layers).foldl (fun x bl => netStep R N.W (one R N) bl.1 bl.2 x) x

variable (M : Transformer)

/-- the rounded embedding of a token -/
def embed (τ : ℕ) : ℕ → ℝ := fun f => R.fl (M.embed τ f)

/-- `z = rmsnorm(embed[token], embedded, one)` -/
def zvec (τ : ℕ) : ℕ → ℝ := fun f => R.fl (rmsnorm M.net.W M.embedded (one R M.net) (embed R M τ) f)

/-- `fl(A @ z)` with `A` rounded -/
def matvec (A : ℕ → ℕ → ℝ) (z : ℕ → ℝ) : ℕ → ℝ :=
  fun d => R.fl (∑ f ∈ range M.net.W, R.fl (A d f) * z f)

/-- the key at `p'` -/
def key (toks : ℕ → ℕ) (p' : ℕ) : ℕ → ℝ :=
  fun d => R.fl (rope (matvec R M M.k (zvec R M (toks p'))) p' d)

/-- the query of head `h` at `p` -/
def query (toks : ℕ → ℕ) (h p : ℕ) : ℕ → ℝ :=
  fun d => R.fl (rope (matvec R M (M.q h) (zvec R M (toks p))) p d)

/-- the score of `p'` for head `h` at `p` -/
def score (toks : ℕ → ℕ) (h p p' : ℕ) : ℝ :=
  R.fl (∑ d ∈ range (2 * PAIRS), query R M toks h p d * key R M toks p' d)

/-- the softmax of the rounded scores, rounded -/
def attn (toks : ℕ → ℕ) (h p p' : ℕ) : ℝ :=
  R.fl (Real.exp (score R M toks h p p') / ∑ p'' ∈ visible p, Real.exp (score R M toks h p p''))

/-- the value at `p'` -/
def value (toks : ℕ → ℕ) (p' : ℕ) : ℕ → ℝ := matvec R M M.v (zvec R M (toks p'))

/-- `fl(a @ values)` for head `h` at `p`, at bit `j` -/
def head (toks : ℕ → ℕ) (h p j : ℕ) : ℝ :=
  R.fl (∑ p' ∈ visible p, attn R M toks h p p' * value R M toks p' j)

/-- The stream at `p` after the heads, added in turn: `x = fl(x + fl(head @ o[h].T))`. -/
def afterAttn (toks : ℕ → ℕ) (p : ℕ) : ℕ → ℝ :=
  (List.range M.heads).foldl
    (fun x h f => R.fl (x f + R.fl (∑ j ∈ range M.bits, R.fl (M.o h f j) * head R M toks h p j)))
    (embed R M (toks p))

/-- The stream at `p` after the last layer. -/
def finalAt (toks : ℕ → ℕ) (p : ℕ) : ℕ → ℝ := final R M.net (afterAttn R M toks p)

/-- The logit of token `τ` at `p`. -/
def logit (toks : ℕ → ℕ) (p τ : ℕ) : ℝ :=
  R.fl (∑ f ∈ range M.net.W,
    norm R M.net.W (one R M.net) (M.net.grid.getLast?.getD fun _ => none) (finalAt R M toks p) f *
      R.fl (M.embed τ f))

end BF16

end

end Reifier
