import ReifierProofs.Interfaces

/-!
# The theorems

For every well-formed program `t` on bits (`IsProgram`), every coding of tokens, every token
sequence and every position where the program's window ends and whose tokens have codes:

* `MainReal`: the transformer compiled from `t` (`transformer t codes`), run in real arithmetic
  with `gelu` and softmax attention, holds each output bit of `t` within `10^-9`, as -1 or 1, and
  the token whose code `t` gives has the largest logit. It needs flat GLUs and weights of
  bounded size (`RealOK`).
* `MainBF16`: run in bf16 (`BF16`), it holds each output bit exactly, and if
  `2 sqrt(width) + bits ≤ 250` the token whose code `t` gives has the largest logit. It needs
  the weights and the gates of the units to be bf16 numbers on bits (`BF16OK`).
-/

namespace Reifier

noncomputable section

/-- **The theorem in real arithmetic.** -/
def MainReal : Prop :=
  ∀ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram → t.RealOK → CodesOK t codes → PosOK t codes toks p →
      let M := transformer t codes
      let x := windowBits t codes toks p
      (∀ i < t.outputs.length, |M.final toks p (1 + i) - (2 * t.outVal x i - 1)| ≤ 1 / 10 ^ 9) ∧
      ∀ τ, CodeIs t codes τ (valAt t codes toks p) →
        ∀ τ' < M.vocab, τ' ≠ τ → M.logit toks p τ' < M.logit toks p τ

/-- **The theorem in bf16.** -/
def MainBF16 : Prop :=
  ∀ (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram → t.BF16OK → CodesOK t codes → PosOK t codes toks p → WidthOK t →
      let M := transformer t codes
      let x := windowBits t codes toks p
      (∀ i < t.outputs.length, BF16.finalAt R M toks p (1 + i) = 2 * t.outVal x i - 1) ∧
      (2 * Real.sqrt (tokenGrid t).width + t.outputs.length ≤ 250 →
        ∀ τ, CodeIs t codes τ (valAt t codes toks p) →
          ∀ τ' < M.vocab, τ' ≠ τ → BF16.logit R M toks p τ' < BF16.logit R M toks p τ)

/-- **The theorem in real arithmetic, to a bound on depth.** For GLUs that are not flat, of
first-order gain at most `σ`: up to the depth `D` where `(1 + σ)^D ≤ 10^4`, the transformer holds
each output bit within `(1 + σ)^D / 10^10`, and gives the right next token. (With `σ = 0` it is
`MainReal` for any depth, with a tighter bound on the GLUs' second-order gain.) -/
def MainRealDeep : Prop :=
  ∀ (σ : ℝ) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram → t.RealCore (10 ^ 5) σ → 0 ≤ σ →
    (1 + σ) ^ ((tokenGrid t).rows.length - 1) ≤ 10 ^ 4 →
    CodesOK t codes → PosOK t codes toks p →
      let M := transformer t codes
      let x := windowBits t codes toks p
      (∀ i < t.outputs.length, |M.final toks p (1 + i) - (2 * t.outVal x i - 1)| ≤
        (1 + σ) ^ ((tokenGrid t).rows.length - 1) / 10 ^ 10) ∧
      ∀ τ, CodeIs t codes τ (valAt t codes toks p) →
        ∀ τ' < M.vocab, τ' ≠ τ → M.logit toks p τ' < M.logit toks p τ

/-- Rounding to bf16 is possible: the theorem in bf16 is not about nothing. -/
def Stmt_rounding_exists : Prop := Nonempty Rounding

end

end Reifier
