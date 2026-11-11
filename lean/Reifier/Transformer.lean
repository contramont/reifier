module

public import Reifier.Emit

@[expose] public section

/-!
# A small transformer with lookback attention: `src/reifier/transformer.py`

A program maps the bits of the last tokens, the oldest token's first, to the bits of the next
token. `codes` gives each token its bits, its code. The residual stream is the grid's rows, with
bits as -1 or 1:

    x = embed[tokens]                    ONE and the token's own code
    x = x + attention(norm(x))           row 0: head h copies the code of the token h back
    x = x + layer(norm(x)), per layer    rows 1, 2, ...: emit's layers, as they are
    logits = norm(x) @ embed.T

Here a stream is held for one position at a time, so `Transformer.run` gives
`[position][depth][feature]` where numpy has `[depth][position][feature]`.
-/

namespace Reifier

variable {K : Type}

/-- `PAIRS = 8`, of dimensions of a query or a key. -/
def PAIRS : ℕ := 8

/-- `CONTEXT = 512`: a token attends to itself and the `CONTEXT - 1` before it. -/
def CONTEXT : ℕ := 512

/-- `MARGIN = 30.0`: a head scores its token `MARGIN` more than any other in `CONTEXT`. -/
def MARGIN : ℝ := 30

/-- `class Transformer` -/
structure Transformer where
  /-- the bits of each token: the items of a dict, whose keys are distinct -/
  codes : List (ℕ × List ℕ)
  /-- `[token][feature]`, tied to the logits -/
  embed : List (List ℝ)
  /-- RoPE turns pair `i` of dimensions by `turns[i]` radians a position -/
  turns : List ℝ
  /-- `[head][dimension][feature]`: the first of each pair, then the second -/
  q : List (List (List ℝ))
  /-- `[dimension][feature]`, the same for every head -/
  k : List (List ℝ)
  /-- `[bit][feature]`, the same for every head -/
  v : List (List ℝ)
  /-- `[head][feature][bit]` -/
  o : List (List (List ℝ))
  /-- the feed-forward layers, and the grid that names every feature -/
  net : Network

/-- `x @ w.T`: the matrix `w`, a list of rows, times the vector `x`. -/
def matVec (w : List (List ℝ)) (x : List ℝ) : List ℝ := w.map (dot · x)

/-- numpy's `max` of a vector that is not empty. -/
noncomputable def maxOf (xs : List ℝ) : ℝ := xs.foldl max (xs.headD 0)

/-- numpy's `argmax`: the first index of the largest. -/
noncomputable def argmax (xs : List ℝ) : ℕ := xs.idxOf (maxOf xs)

/-- `rope(x, pos, turns)` at one position: `x[dimension]` with each pair of dimensions
`(i, i + half)` turned by `pos * turns[i]`. -/
noncomputable def rope (x : List ℝ) (pos : ℕ) (turns : List ℝ) : List ℝ :=
  let a := x.take (x.length / 2)
  let b := x.drop (x.length / 2)
  let angle := turns.map ((pos : ℝ) * ·)
  List.zipWith₃ (fun a b θ => a * Real.cos θ - b * Real.sin θ) a b angle
    ++ List.zipWith₃ (fun a b θ => b * Real.cos θ + a * Real.sin θ) a b angle

/-- `lookback_query(h, turns)`: the query of a head that attends to the token `h` back. Every
key is 1 in the first of each pair, and the query is that key turned back `h` positions. -/
noncomputable def lookbackQuery (h : ℕ) (turns : List ℝ) : List ℝ :=
  turns.map (fun t => Real.cos (t * h)) ++ turns.map (fun t => -Real.sin (t * h))

/-- `runner_up(turns, n)`: the largest `mean(cos(e turns))` over offsets `0 < e < n`. -/
noncomputable def runnerUp (turns : List ℝ) (n : ℕ) : ℝ :=
  maxOf ((List.range' 1 (n - 1)).map fun e =>
    (turns.map fun t => Real.cos (e * t)).sum / turns.length)

/-- The first half of `Transformer.run`: the stream of every position after the attention,
`x = embed[tokens]` and then `x = x + attention(norm(x))`, a head at a time. -/
noncomputable def Transformer.attend (m : Transformer) (tokens : List ℕ) : List (List ℝ) :=
  let x := tokens.map fun token => m.embed.getD token []
  -- an embedding holds ONE and the code, no more
  let embedded : Row := (m.net.grid.headD []).zipIdx.map fun s => if s.2 ≤ m.v.length then s.1 else none
  let z := x.map fun x => rmsnorm x embedded m.net.one
  let keys := z.zipIdx.map fun z => rope (matVec m.k z.1) z.2 m.turns
  let values := z.map (matVec m.v)
  (m.q.zip m.o).foldl
    (fun x qo => (x.zip z).zipIdx.map fun xz =>
      let pos := xz.2
      let scores := keys.map (dot (rope (matVec qo.1 xz.1.2) pos m.turns))
      -- a token attends to itself and the CONTEXT - 1 before it
      let seen := fun j : ℕ => decide (j ≤ pos) && decide (pos - j < CONTEXT)
      let top := maxOf ((scores.zipIdx.filter fun s => seen s.2).map (·.1))
      let a := scores.zipIdx.map fun s => if seen s.2 then Real.exp (s.1 - top) else 0
      let mix := (List.range m.v.length).map fun i =>
        ((a.zip values).map fun av => av.1 / a.sum * av.2.getD i 0).sum
      List.zipWith (· + ·) xz.1.1 (matVec qo.2 mix))
    x

/-- `Transformer.run(tokens)`: the stream `[position][depth][feature]` at every depth of the
grid, and the logits `[position][token]`. -/
noncomputable def Transformer.run (m : Transformer) (tokens : List ℕ) :
    List (List (List ℝ)) × List (List ℝ) :=
  let xs := (m.attend tokens).map m.net.run
  (xs, xs.map fun streams =>
    matVec m.embed (rmsnorm (streams.getLastD []) (m.net.grid.getLastD []) m.net.one))

/-- The greedy next token: `int(self.run(tokens)[1][-1].argmax())`. -/
noncomputable def Transformer.next (m : Transformer) (tokens : List ℕ) : ℕ :=
  argmax ((m.run tokens).2.getLastD [])

/-- `Transformer.generate(tokens, n)`: `tokens` and the `n` greedy next ones. -/
noncomputable def Transformer.generate (m : Transformer) (tokens : List ℕ) (n : ℕ) : List ℕ :=
  (fun tokens => tokens ++ [m.next tokens])^[n] tokens

/-- `f"code[{i}]"`: the name of bit `i` of a token's own code. -/
def codeName (i : ℕ) : String := s!"code[{i}]"

/-- `token_grid(t)`: the grid of a program on tokens, held: `layout`'s, with features 1, 2, ...
for `code[0]`, `code[1]`, ..., the bits of the token's own code. Output `i` is moved from the
feature that `layout` gives it to feature `1 + i`, so that the layer that makes it writes it
over `code[i]`, where the logits read it. -/
def tokenGrid (t : Trace K) : Grid :=
  let code := (List.range t.outputs.length).map fun i => some (codeName i)
  let grid : Grid := (layout t).map fun row => some ONE :: code ++ row.drop 1
  held (t.outputs.zipIdx.foldl
    (fun grid yi => grid.map fun row =>
      if row.contains (some yi.1) then
        (row.set (row.idxOf (some yi.1)) none).set (1 + yi.2) (some yi.1)
      else row)
    grid)

/-- `transformer(t, codes)`: `t`, a program from the bits of the last tokens, the oldest
token's first, to the bits of the next, as a transformer on the tokens of `codes`. -/
noncomputable def transformer (t : Trace ℝ) (codes : List (ℕ × List ℕ)) : Transformer :=
  let bits := t.outputs.length
  let grid := tokenGrid t
  let net := emit t grid .geglu true true
  let width := (grid.headD []).length
  let window := t.inputs.length / bits  -- the tokens that t reads
  let embed := (List.range ((codes.map (·.1)).foldl max 0 + 1)).map fun token =>
    match codes.lookup token with
    | some code =>
      let own := net.one :: code.map fun c => 2 * (c : ℝ) - 1
      own ++ List.replicate (width - own.length) 0
    | none => List.replicate width 0
  let turns := (List.range PAIRS).map fun i => (10000 : ℝ) ^ (-((i : ℝ) / PAIRS))
  -- its own token scores scale * PAIRS
  let scale := MARGIN / (PAIRS * (1 - runnerUp turns CONTEXT))
  { codes := codes
    embed := embed
    turns := turns
    -- head h: the code of the token h back, into its inputs' features
    q := (List.range window).map fun h =>
      (lookbackQuery h turns).map fun c => (scale * c) :: List.replicate (width - 1) 0
    -- a key is 1 in the first of each pair: it reads ONE, as a query does
    k := (List.range (2 * PAIRS)).map fun d =>
      (if d < PAIRS then 1 else 0) :: List.replicate (width - 1) 0
    -- a value is the code
    v := (List.range bits).map fun i =>
      (List.range width).map fun f => if f = 1 + i then 1 else 0
    -- the program reads the oldest token first
    o := (List.range window).map fun h =>
      (List.range width).map fun f => (List.range bits).map fun i =>
        if ((t.inputs[(window - 1 - h) * bits + i]?).map fun x => (grid.headD []).idxOf (some x))
            = some f then 1 else 0
    net := net }

end Reifier
