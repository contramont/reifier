import ReifierProofs.Spec

/-!
# The proof, as statements

Each `Stmt_*` is one step of the proof, stated as a proposition. Each is proved in a file of
its own under `ReifierProofs/Proofs/`, which may assume other `Stmt_*` as hypotheses; `Final.lean`
puts them together. Nothing here is assumed: a `Stmt_*` is a definition, not an axiom.
-/

namespace Reifier

open Finset

noncomputable section

/-- The stream that `row` describes: `ONE` as `one`, a name of value `v` as `2 v - 1` (a bit held
as -1 or 1), and 0 where the row holds nothing (Python `Network.stream`, with norms). -/
def ideal (one : ℝ) (val : Name → ℝ) (row : Row) : ℕ → ℝ := fun f =>
  match row f with
  | some .one => one
  | some n => 2 * val n - 1
  | none => 0

/-- The value of each name at position `p`: a signal's in the trace, on the bits of the window
that ends at `p`; `code[i]`, bit `i` of the token at `p`. -/
def valAt (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) : Name → ℝ
  | .one => 0
  | .sig s => t.values (t.env (windowBits t codes toks p)) s
  | .code i => if ((codes.lookup (toks p)).getD []).getD i false then 1 else 0

/-- `val` gives the signals their values in `t` on some input bits, and the code bits bits. -/
def ValOK (t : Trace) (val : Name → ℝ) : Prop :=
  (∃ x : ℕ → Bool, ∀ s, val (.sig s) = t.values (t.env x) s) ∧ ∀ i, IsBit (val (.code i))

/-- A row of a grid for `t`, of width `W`: `ONE` in feature 0 and nowhere else, nothing beyond
`W`, no name twice, and each name one with a value: a code bit, an input, or an MLP's output. -/
structure RowOK (t : Trace) (W : ℕ) (r : Row) : Prop where
  one_zero : r 0 = some .one
  one_only : ∀ f, r f = some .one → f = 0
  none_beyond : ∀ f, W ≤ f → r f = none
  nodup : ∀ f g, r f = r g → (r f).isSome → f = g
  names : ∀ f n, r f = some n →
    n = .one ∨ (∃ i, n = .code i ∧ i < t.outputs.length) ∨
      (∃ s, n = .sig s ∧ (s ∈ t.inputs ∨ ∃ m ∈ t.mlps, s ∈ m.outs))

/-- Two rows of a grid, `before` and the `row` after it: a cell that holds a name keeps one, and
a cell that changes gets an output of an MLP that is new here, which reads only what `before`
holds and whose outputs are all in `row` and none in `before`. -/
structure PairOK (t : Trace) (W : ℕ) (before row : Row) : Prop where
  before_ok : RowOK t W before
  row_ok : RowOK t W row
  held : ∀ f, (before f).isSome → (row f).isSome
  changed : ∀ f, row f ≠ before f →
    ∃ m ∈ t.mlps, mlpIsNew W before row m = true ∧ ∃ y ∈ m.outs, row f = some (.sig y)
  new_ins : ∀ m ∈ t.mlps, mlpIsNew W before row m = true → ∀ x ∈ m.ins,
    inRow W before (.sig x) = true
  new_outs : ∀ m ∈ t.mlps, mlpIsNew W before row m = true → ∀ y ∈ m.outs,
    inRow W row (.sig y) = true ∧ inRow W before (.sig y) = false

/-- Row 0 of a token grid: `ONE`, the code bits at `1 .. bits`, the inputs, and nothing else. -/
structure Row0OK (t : Trace) (W : ℕ) (r : Row) : Prop where
  width : t.outputs.length + 1 ≤ W
  code : ∀ i < t.outputs.length, r (1 + i) = some (.code i)
  inputs : ∀ x ∈ t.inputs, inRow W r (.sig x) = true
  names : ∀ f n, r f = some n →
    n = .one ∨ (∃ i, n = .code i ∧ i < t.outputs.length) ∨ (∃ x ∈ t.inputs, n = .sig x)

/-- What the proof needs of the token grid. -/
structure GridOK (t : Trace) (g : Grid) : Prop where
  rows_ne : g.rows ≠ []
  rows_ok : ∀ r ∈ g.rows, RowOK t g.width r
  pairs : ∀ (d : ℕ) (hd : d + 1 < g.rows.length),
    PairOK t g.width (g.rows[d]'(by omega)) (g.rows[d + 1])
  row0 : Row0OK t g.width (g.rows.head rows_ne)
  last : ∀ i < t.outputs.length,
    (g.rows.getLast rows_ne) (1 + i) = some (.sig (t.outputs.getD i 0))

/-! ## The grid -/

/-- `token_grid` gives a grid that the layers can follow. -/
def Stmt_grid : Prop := ∀ t : Trace, t.IsProgram → GridOK t (tokenGrid t)

/-! ## The trace -/

/-- The value of an MLP's output in the trace is what the MLP gives on the values of its
inputs, and an input's is the bit it is given. -/
def Stmt_trace : Prop :=
  ∀ (t : Trace) (x : Sig → ℝ), t.WellFormed →
    (∀ s ∈ t.inputs, t.values x s = x s) ∧
    ∀ m ∈ t.mlps, ∀ i < m.outs.length,
      t.values x (m.outs.getD i 0) = m.out (m.read (t.values x)) i

/-! ## A layer, unit by unit -/

/-- The gate of unit `j` of a placed MLP, on the stream `z`. -/
def gateDot (W : ℕ) (before : Row) (m : MLP) (j : ℕ) (z : ℕ → ℝ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.compiled.g j k : ℝ) * z (featIn W before (.sig (m.ins.getD k 0)))
    + m.compiled.gb j * z 0

/-- The value of unit `j` of a placed MLP, on the stream `z`. -/
def valDot (W : ℕ) (before : Row) (m : MLP) (j : ℕ) (z : ℕ → ℝ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.compiled.v j k : ℝ) * z (featIn W before (.sig (m.ins.getD k 0)))
    + m.compiled.vb j * z 0

/-- The layer at feature `f`, with each weight evaluated by `φ` and each unit by `ψ`: the sum
over its units of `φ (u.out f) * ψ u`. In real arithmetic `φ` is the cast and `ψ u` is
`act(gate . z) * (val . z)`; in bf16 both are rounded. -/
def unitSum (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (us : List Hidden) (f : ℕ) : ℝ :=
  (us.map fun u => φ (u.out f) * ψ u).sum

/-- What a layer writes into each feature, unit by unit: nothing into one that keeps its name;
into one where output `i` of a new MLP `m` goes, the unit `ONE`'s -1, the units of `m` with their
weights into output `i`, and the erasure of what the feature held. -/
def Stmt_decomp : Prop :=
  ∀ (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (t : Trace) (W : ℕ) (before row : Row),
    φ 0 = 0 → t.WellFormed → PairOK t W before row →
    (∀ f, row f = before f → unitSum φ ψ (layerUnits W t.mlps before row) f = 0) ∧
    ∀ (f : ℕ) (m : MLP) (i : ℕ), m ∈ t.mlps → mlpIsNew W before row m = true →
      i < m.outs.length → row f = some (.sig (m.outs.getD i 0)) →
        unitSum φ ψ (layerUnits W t.mlps before row) f =
          φ (-1) * ψ (oneUnit W before row)
          + ∑ j ∈ range m.compiled.n,
              φ (m.compiled.wo i j) * ψ ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩)
          + (if (before f).isSome then φ (-1) * ψ (eraseUnit f) else 0)

/-- The weights of each unit, feature by feature, and so its gate and value on a stream. A new
MLP's unit `j` has its bias on `ONE`, its weight `gOn j s` on the feature of each signal `s` of
`before`, and nothing elsewhere. -/
def Stmt_unit_dots : Prop :=
  (∀ (W : ℕ) (before row : Row) (z : ℕ → ℝ), 0 < W →
    dot W (oneUnit W before row).gate z = 32 * z 0 ∧ dot W (oneUnit W before row).val z = z 0 / 32) ∧
  (∀ (W f : ℕ) (z : ℕ → ℝ), 1 ≤ f → f < W →
    dot W (eraseUnit f).gate z = 32 * z 0 ∧ dot W (eraseUnit f).val z = z f / 32) ∧
  ∀ (t : Trace) (W : ℕ) (before row : Row) (m : MLP) (j : ℕ),
    t.WellFormed → PairOK t W before row → m ∈ t.mlps → mlpIsNew W before row m = true →
    j < m.compiled.n →
      let u := (mlpUnits W before row m).getD j ⟨0, 0, 0⟩
      (∀ z, dot W u.gate z = gateDot W before m j z ∧ dot W u.val z = valDot W before m j z) ∧
      (u.gate 0 = m.compiled.gb j ∧ u.val 0 = m.compiled.vb j) ∧
      (∀ g s, before g = some (.sig s) → u.gate g = m.compiled.gOn j s ∧ u.val g = m.compiled.vOn j s) ∧
      (∀ g, 1 ≤ g → (∀ s, before g ≠ some (.sig s)) → u.gate g = 0 ∧ u.val g = 0)

/-! ## A compiled MLP -/

/-- What `signed(convert(m, "geglu"))` computes, against `m.toReglu`: on a stream that holds the
inputs `w` as `c (2 w - 1)` and `ONE` as `c`, its gates are `32 c` times those of the reglu
form, its values `c / 32` times, and its outputs twice. And the reglu form computes `m` on bits,
with every gate 0 or at least 1/2 from 0 (at least 1/2 for a step, whose two units `u+` and `u-`
are on the same side of 0), a `relu` or GLU gate being the MLP's own sum. -/
def Stmt_compiled : Prop :=
  (∀ (m : MLP) (c : ℝ) (w : ℕ → ℝ) (j : ℕ),
    (∑ k ∈ range m.ins.length, (m.compiled.g j k : ℝ) * (c * (2 * w k - 1)) + m.compiled.gb j * c
      = 32 * c * (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * w k + m.toReglu.gb j)) ∧
    (∑ k ∈ range m.ins.length, (m.compiled.v j k : ℝ) * (c * (2 * w k - 1)) + m.compiled.vb j * c
      = c / 32 * (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * w k + m.toReglu.vb j))) ∧
  (∀ (m : MLP) (i j : ℕ), m.compiled.wo i j = 2 * m.toReglu.wo i j) ∧
  (∀ m : MLP, m.compiled.n = m.toReglu.n ∧ m.compiled.ins = m.ins ∧ m.compiled.outs = m.outs) ∧
  ∀ (t : Trace) (x : ℕ → Bool) (m : MLP), t.WellFormed → m ∈ t.mlps →
    let u := t.inVals x m
    (∀ i, m.out u i = ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) *
      (relu (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j) *
        (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * u k + m.toReglu.vb j))) ∧
    (∀ j < m.toReglu.n,
      (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j) = 0 ∨
        1 / 2 ≤ |∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j|) ∧
    (m.act = .step →
      (∀ j < m.toReglu.n,
        1 / 2 ≤ |∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j|) ∧
      ∀ j < m.hidden,
        (0 < ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j ↔
          0 < ∑ k ∈ range m.ins.length, (m.toReglu.g (m.hidden + j) k : ℝ) * u k +
            m.toReglu.gb (m.hidden + j))) ∧
    (m.act ≠ .step → ∀ j < m.toReglu.n,
      ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j = m.pre u j)

/-! ## The norm -/

/-- On the stream a row describes, the norm gives that row's stream with `ONE` as 1. -/
def Stmt_norm_exact : Prop :=
  ∀ (t : Trace) (W : ℕ) (row : Row) (val : Name → ℝ) (one : ℝ),
    RowOK t W row → ValOK t val → t.WellFormed → 0 < one → 0 < W →
      ∀ f, rmsnorm W row one (ideal one val row) f = ideal 1 val row f

/-- On a stream at most `E` off the one a row describes, with `ONE` exact at `2 sqrt W`, the norm
scales every feature but `ONE` by one `c`, and `ONE` to `c`, with `c^2` at most `0.41 E` off 1. -/
def Stmt_norm_close : Prop :=
  ∀ (t : Trace) (W : ℕ) (row : Row) (val : Name → ℝ) (x : ℕ → ℝ) (E : ℝ),
    RowOK t W row → ValOK t val → t.WellFormed → 0 < W → 0 ≤ E → E ≤ 1 / 10 ^ 6 →
    x 0 = 2 * Real.sqrt W →
    (∀ f, 1 ≤ f → f < W → |x f - ideal (2 * Real.sqrt W) val row f| ≤ E) →
      ∃ c : ℝ, 0 < c ∧ |c ^ 2 - 1| ≤ 41 / 100 * E ∧
        ∀ f, rmsnorm W row (2 * Real.sqrt W) x f = if f = 0 then c else c * x f

/-! ## gelu -/

/-- `gelu` is `relu` but for `10^-20`, at least 8 from 0. -/
def Stmt_gelu : Prop := ∀ z : ℝ, 8 ≤ |z| → |gelu z - relu z| ≤ 1 / 10 ^ 20

/-! ## Rounding to bf16 -/

/-- What the proof in bf16 needs of rounding: it keeps what bf16 holds; small integers times
powers of two are bf16 numbers; it rounds `gelu` of a gate of bf16 at least 16 from 0 to `relu`
of it; it is monotone; a real within `2^-9` of `±1`, or of 0 within `2^-135`, rounds to it; and
it is off by at most `2^-8` of what it rounds, and `2^-134`. -/
def Stmt_round : Prop :=
  (∀ (R : Rounding) (x : ℝ), IsBF16 x → R.fl x = x) ∧
  (∀ x : ℝ, IsBF16 x → |x| ≤ maxBF16) ∧
  (∀ (n : ℤ) (k : ℕ), |n| ≤ 256 → k ≤ 100 → IsBF16 ((n : ℝ) / 2 ^ k)) ∧
  (∀ (n : ℤ) (k : ℕ), |n| ≤ 256 → k ≤ 100 → IsBF16 ((n : ℝ) * 2 ^ k)) ∧
  (∀ (R : Rounding) (G : ℝ), IsBF16 G → 16 ≤ |G| → R.fl (gelu G) = relu G) ∧
  (∀ (R : Rounding) (x y : ℝ), |x| ≤ maxBF16 → |y| ≤ maxBF16 → x ≤ y → R.fl x ≤ R.fl y) ∧
  (∀ (R : Rounding) (x : ℝ) (s : ℝ), (s = 1 ∨ s = -1) → |x - s| < 1 / 2 ^ 9 → R.fl x = s) ∧
  (∀ (R : Rounding) (x : ℝ), |x| < 1 / 2 ^ 135 → R.fl x = 0) ∧
  (∀ (R : Rounding) (x : ℝ), |x| ≤ maxBF16 → |R.fl x - x| ≤ |x| / 2 ^ 8 + 1 / 2 ^ 134)

/-! ## A layer -/

/-- The invariant of the forward pass in real arithmetic: `ONE` exact, nothing where the row holds
nothing, and every feature at most `E` from the stream the row describes. -/
structure Close (W : ℕ) (E one : ℝ) (val : Name → ℝ) (row : Row) (x : ℕ → ℝ) : Prop where
  one_eq : x 0 = one
  none_eq : ∀ f, 1 ≤ f → row f = none → x f = 0
  near : ∀ f, 1 ≤ f → f < W → |x f - ideal one val row f| ≤ E

/-- One layer, in real arithmetic with `gelu`: a stream `10^-9` close to its row's gives one as
close to the next row's. -/
def Stmt_layer_real : Prop :=
  ∀ (t : Trace) (W : ℕ) (before row : Row) (val : Name → ℝ) (x : ℕ → ℝ),
    t.WellFormed → t.RealOK → PairOK t W before row → ValOK t val → 0 < W →
    Close W (1 / 10 ^ 9) (2 * Real.sqrt W) val before x →
      Close W (1 / 10 ^ 9) (2 * Real.sqrt W) val row
        (netStep gelu W (2 * Real.sqrt W) before (layerUnits W t.mlps before row) x)

/-- One layer, in real arithmetic with `gelu`, for an error `E` between `10^-10` and `10^-6`: a
stream `E` close to its row's gives one `(1 + σ) E` close to the next row's, where `σ` bounds the
first-order gain of every GLU (0 if they are flat) and `G` their second-order gain, `G E ≤ 1/10`,
and `σ E ≤ 1/100` (the norm's own error, `0.41 E`, scales what the GLUs pass on). -/
def Stmt_layer_gain : Prop :=
  ∀ (t : Trace) (W : ℕ) (before row : Row) (val : Name → ℝ) (x : ℕ → ℝ) (E G σ : ℝ),
    t.WellFormed → t.RealCore G σ → PairOK t W before row → ValOK t val → 0 < W →
    1 / 10 ^ 10 ≤ E → E ≤ 1 / 10 ^ 6 → 0 ≤ G → G * E ≤ 1 / 10 → 0 ≤ σ → σ * E ≤ 1 / 100 →
    Close W E (2 * Real.sqrt W) val before x →
      Close W ((1 + σ) * E) (2 * Real.sqrt W) val row
        (netStep gelu W (2 * Real.sqrt W) before (layerUnits W t.mlps before row) x)

/-- One layer, in bf16: the stream a row describes gives exactly the next row's. -/
def Stmt_layer_bf16 : Prop :=
  ∀ (R : Rounding) (t : Trace) (W : ℕ) (before row : Row) (val : Name → ℝ) (one : ℝ),
    t.WellFormed → t.BF16OK → PairOK t W before row → ValOK t val → 0 < W →
    IsBF16 one → 1 ≤ one →
      ∀ f, BF16.netStep R W one before (layerUnits W t.mlps before row) (ideal one val before) f =
        ideal one val row f

/-! ## The attention -/

/-- The hypotheses on a position: the window ends there, and its tokens have codes (the token at
`p` too, even when the program reads no token). -/
def PosOK (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) : Prop :=
  t.window ≤ p + 1 ∧ ∀ h ≤ t.window - 1, (codes.lookup (toks (p - h))).isSome

/-- In real arithmetic, the stream after the attention is `10^-10` close to row 0's. -/
def Stmt_attn_real : Prop :=
  ∀ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram → CodesOK t codes → PosOK t codes toks p →
      let M := transformer t codes
      Close M.net.W (1 / 10 ^ 10) M.net.one (valAt t codes toks p)
        ((tokenGrid t).rows.headD fun _ => none) (M.afterAttn toks p)

/-- PATCH (CE-11): `ONE`, `2 sqrt(width)`, is a finite bf16 number. -/
def WidthOK (t : Trace) : Prop := 2 * Real.sqrt (tokenGrid t).width ≤ maxBF16

/-- In bf16, the stream after the attention is exactly row 0's. -/
def Stmt_attn_bf16 : Prop :=
  ∀ (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram → CodesOK t codes → PosOK t codes toks p → WidthOK t →
      let M := transformer t codes
      ∀ f, BF16.afterAttn R M toks p f =
        ideal (BF16.one R M.net) (valAt t codes toks p) ((tokenGrid t).rows.headD fun _ => none) f

/-! ## The logits -/

/-- The logit of `τ` on a final stream `x`, in real arithmetic. -/
def logitOf (M : Transformer) (x : ℕ → ℝ) (τ : ℕ) : ℝ :=
  ∑ f ∈ range M.net.W, rmsnorm M.net.W (M.net.grid.getLast?.getD fun _ => none) M.net.one x f
    * M.embed τ f

/-- The logit of `τ` on a final stream `x`, in bf16. -/
def blogitOf (R : Rounding) (M : Transformer) (x : ℕ → ℝ) (τ : ℕ) : ℝ :=
  R.fl (∑ f ∈ range M.net.W,
    BF16.norm R M.net.W (BF16.one R M.net) (M.net.grid.getLast?.getD fun _ => none) x f *
      R.fl (M.embed τ f))

/-- The code of `τ` is what the trace gives. -/
def CodeIs (t : Trace) (codes : List (ℕ × List Bool)) (τ : ℕ) (val : Name → ℝ) : Prop :=
  ∃ c, codes.lookup τ = some c ∧
    ∀ i < t.outputs.length, (if c.getD i false then (1 : ℝ) else 0) = val (.sig (t.outputs.getD i 0))

/-- In real arithmetic: on a final stream `10^-9` close to the last row's, the token whose code
the trace gives has the largest logit. -/
def Stmt_logit_real : Prop :=
  ∀ (t : Trace) (codes : List (ℕ × List Bool)) (val : Name → ℝ) (x : ℕ → ℝ) (τ : ℕ),
    t.IsProgram → CodesOK t codes → ValOK t val →
      let M := transformer t codes
      Close M.net.W (1 / 10 ^ 9) M.net.one val ((tokenGrid t).rows.getLast?.getD fun _ => none) x →
      CodeIs t codes τ val → ∀ τ' < M.vocab, τ' ≠ τ → logitOf M x τ' < logitOf M x τ

/-- In real arithmetic: on a final stream `E ≤ 10^-6` close to the last row's, the token whose code
the trace gives has the largest logit. -/
def Stmt_logit_real_E : Prop :=
  ∀ (t : Trace) (codes : List (ℕ × List Bool)) (val : Name → ℝ) (x : ℕ → ℝ) (τ : ℕ) (E : ℝ),
    t.IsProgram → CodesOK t codes → ValOK t val → 0 ≤ E → E ≤ 1 / 10 ^ 6 →
      let M := transformer t codes
      Close M.net.W E M.net.one val ((tokenGrid t).rows.getLast?.getD fun _ => none) x →
      CodeIs t codes τ val → ∀ τ' < M.vocab, τ' ≠ τ → logitOf M x τ' < logitOf M x τ

/-- In bf16: on the final stream the last row describes, the token whose code the trace gives
has the largest logit, if `2 sqrt(width) + bits ≤ 250`. -/
def Stmt_logit_bf16 : Prop :=
  ∀ (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (val : Name → ℝ) (τ : ℕ),
    t.IsProgram → CodesOK t codes → ValOK t val →
    2 * Real.sqrt (tokenGrid t).width + t.outputs.length ≤ 250 →
      let M := transformer t codes
      let x := ideal (BF16.one R M.net) val ((tokenGrid t).rows.getLast?.getD fun _ => none)
      CodeIs t codes τ val → ∀ τ' < M.vocab, τ' ≠ τ → blogitOf R M x τ' < blogitOf R M x τ

end

end Reifier
