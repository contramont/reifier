import ReifierProofs.Interfaces

namespace Reifier
namespace AttnBF16Err
open Finset

lemma cs2 (a b c d X Y : ℝ) (hX : 0 ≤ X) (hY : 0 ≤ Y) (h1 : a ^ 2 + b ^ 2 ≤ X ^ 2)
    (h2 : c ^ 2 + d ^ 2 ≤ Y ^ 2) : |a| * |c| + |b| * |d| ≤ X * Y := by
  have key : (|a| * |c| + |b| * |d|) ^ 2 ≤ (X * Y) ^ 2 := by
    have : (|a| * |c| + |b| * |d|) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2) := by
      nlinarith [sq_nonneg (|a| * |d| - |b| * |c|), sq_abs a, sq_abs b, sq_abs c, sq_abs d]
    calc _ ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2) := this
      _ ≤ X ^ 2 * Y ^ 2 := mul_le_mul h1 h2 (by positivity) (by positivity)
      _ = (X * Y) ^ 2 := by ring
  exact (pow_le_pow_iff_left₀ (by positivity) (by positivity) (by norm_num)).mp key

lemma abs_le_of_sq {a b X : ℝ} (hX : 0 ≤ X) (h : a ^ 2 + b ^ 2 ≤ X ^ 2) : |a| ≤ X := by
  have : a ^ 2 ≤ X ^ 2 := by nlinarith [sq_nonneg b]
  exact abs_le_of_sq_le_sq' this hX |> fun h => abs_le.mpr h

/-- One pair of dimensions of the score: the computed products against the exact ones. -/
lemma pair_core (A cα sα cβ sβ cγ sγ q1 q2 Q1 Q2 K1 K2 η : ℝ)
    (hA : 0 ≤ A) (hA' : A ≤ 62.5) (hη0 : 0 ≤ η) (hηs : η ≤ 1 / 2 ^ 130)
    (hα : cα ^ 2 + sα ^ 2 = 1) (hβ : cβ ^ 2 + sβ ^ 2 = 1) (hγ : cγ ^ 2 + sγ ^ 2 = 1)
    (hq1 : |q1 - A * cα| ≤ |A * cα| / 2 ^ 8 + η)
    (hq2 : |q2 - A * -sα| ≤ |A * -sα| / 2 ^ 8 + η)
    (hQ1 : |Q1 - (q1 * cβ - q2 * sβ)| ≤ |q1 * cβ - q2 * sβ| / 2 ^ 8 + η)
    (hQ2 : |Q2 - (q2 * cβ + q1 * sβ)| ≤ |q2 * cβ + q1 * sβ| / 2 ^ 8 + η)
    (hK1 : |K1 - cγ| ≤ |cγ| / 2 ^ 8 + η)
    (hK2 : |K2 - sγ| ≤ |sγ| / 2 ^ 8 + η) :
    |Q1 * K1 + Q2 * K2 - A * ((cα * cβ + sα * sβ) * cγ + (cα * sβ - sα * cβ) * sγ)|
      ≤ A * (3.02 / 256) + 1 / 2 ^ 120 := by
  set e1 := q1 - A * cα with he1
  set e2 := q2 + A * sα with he2
  have he1b : |e1| ≤ A * |cα| / 256 + η := by
    rw [abs_mul, abs_of_nonneg hA] at hq1; norm_num at hq1 ⊢; linarith
  have he2b : |e2| ≤ A * |sα| / 256 + η := by
    have : q2 - A * -sα = e2 := by rw [he2]; ring
    rw [this, abs_mul, abs_of_nonneg hA, abs_neg] at hq2; norm_num at hq2 ⊢; linarith
  set R1 := A * (cα * cβ + sα * sβ) with hR1
  set R2 := A * (cα * sβ - sα * cβ) with hR2
  set f1 := e1 * cβ - e2 * sβ with hf1
  set f2 := e2 * cβ + e1 * sβ with hf2
  have hr1 : q1 * cβ - q2 * sβ = R1 + f1 := by rw [hR1, hf1, he1, he2]; ring
  have hr2 : q2 * cβ + q1 * sβ = R2 + f2 := by rw [hR2, hf2, he1, he2]; ring
  rw [hr1] at hQ1; rw [hr2] at hQ2
  set g1 := Q1 - (R1 + f1) with hg1
  set g2 := Q2 - (R2 + f2) with hg2
  set k1 := K1 - cγ with hk1
  set k2 := K2 - sγ with hk2
  clear_value e1 e2 R1 R2 f1 f2 g1 g2 k1 k2
  -- the exact pair
  have hRR : R1 ^ 2 + R2 ^ 2 ≤ A ^ 2 := by
    have : R1 ^ 2 + R2 ^ 2 = A ^ 2 * ((cα ^ 2 + sα ^ 2) * (cβ ^ 2 + sβ ^ 2)) := by
      rw [hR1, hR2]; ring
    rw [this, hα, hβ]; simp
  have hcα : |cα| ≤ 1 := abs_le_of_sq (by norm_num) (by rw [hα]; norm_num)
  have hsα : |sα| ≤ 1 := abs_le_of_sq (by norm_num) (by rw [add_comm, hα]; norm_num)
  have hcγ : |cγ| ≤ 1 := abs_le_of_sq (by norm_num) (by rw [hγ]; norm_num)
  have hsγ : |sγ| ≤ 1 := abs_le_of_sq (by norm_num) (by rw [add_comm, hγ]; norm_num)
  have hR1a : |R1| ≤ A := abs_le_of_sq hA hRR
  have hR2a : |R2| ≤ A := abs_le_of_sq (b := R1) hA (by linarith)
  -- the error of the query after the rotation
  obtain ⟨F, hF⟩ : ∃ F : ℝ, F = A / 256 + 2 * η := ⟨_, rfl⟩
  have hff : f1 ^ 2 + f2 ^ 2 ≤ F ^ 2 := by
    have h1 : f1 ^ 2 + f2 ^ 2 = (e1 ^ 2 + e2 ^ 2) * (cβ ^ 2 + sβ ^ 2) := by rw [hf1, hf2]; ring
    rw [h1, hβ, mul_one]
    have a1 : e1 ^ 2 ≤ (A * |cα| / 256 + η) ^ 2 := by
      rw [← sq_abs e1]; exact pow_le_pow_left₀ (abs_nonneg _) he1b 2
    have a2 : e2 ^ 2 ≤ (A * |sα| / 256 + η) ^ 2 := by
      rw [← sq_abs e2]; exact pow_le_pow_left₀ (abs_nonneg _) he2b 2
    have a3 : |cα| ^ 2 + |sα| ^ 2 = 1 := by rw [sq_abs, sq_abs, hα]
    have hAη : 0 ≤ A * η := mul_nonneg hA hη0
    have ex : (A * |cα| / 256 + η) ^ 2 + (A * |sα| / 256 + η) ^ 2
        = A ^ 2 * (|cα| ^ 2 + |sα| ^ 2) / 65536 + 2 * (A * η) * (|cα| + |sα|) / 256 + 2 * η ^ 2 := by
      ring
    rw [a3] at ex
    have m : (A * η) * (|cα| + |sα|) ≤ (A * η) * 2 := mul_le_mul_of_nonneg_left (by linarith) hAη
    have eF : F ^ 2 = A ^ 2 / 65536 + 4 * (A * η) / 256 + 4 * η ^ 2 := by rw [hF]; ring
    have : 0 ≤ η ^ 2 := sq_nonneg η
    linarith
  have hF0 : 0 ≤ F := by rw [hF]; positivity
  -- the key
  obtain ⟨κ, hκ⟩ : ∃ κ : ℝ, κ = 257 / 256 + 2 * η := ⟨_, rfl⟩
  have hκ0 : 0 ≤ κ := by rw [hκ]; positivity
  have hK1a : |K1| ≤ |cγ| * (257 / 256) + η := by
    have : K1 = cγ + k1 := by rw [hk1]; ring
    rw [this]; have := abs_add_le cγ k1; norm_num at hK1 ⊢; linarith
  have hK2a : |K2| ≤ |sγ| * (257 / 256) + η := by
    have : K2 = sγ + k2 := by rw [hk2]; ring
    rw [this]; have := abs_add_le sγ k2; norm_num at hK2 ⊢; linarith
  have hKK : K1 ^ 2 + K2 ^ 2 ≤ κ ^ 2 := by
    have a1 : K1 ^ 2 ≤ (|cγ| * (257 / 256) + η) ^ 2 := by
      rw [← sq_abs K1]; exact pow_le_pow_left₀ (abs_nonneg _) hK1a 2
    have a2 : K2 ^ 2 ≤ (|sγ| * (257 / 256) + η) ^ 2 := by
      rw [← sq_abs K2]; exact pow_le_pow_left₀ (abs_nonneg _) hK2a 2
    have a3 : |cγ| ^ 2 + |sγ| ^ 2 = 1 := by rw [sq_abs, sq_abs, hγ]
    have ex : (|cγ| * (257 / 256) + η) ^ 2 + (|sγ| * (257 / 256) + η) ^ 2
        = (|cγ| ^ 2 + |sγ| ^ 2) * (257 / 256) ^ 2 + 2 * η * (|cγ| + |sγ|) * (257 / 256) + 2 * η ^ 2 := by
      ring
    rw [a3] at ex
    have m : η * (|cγ| + |sγ|) ≤ η * 2 := mul_le_mul_of_nonneg_left (by linarith) hη0
    have eκ : κ ^ 2 = (257 / 256) ^ 2 + 4 * η * (257 / 256) + 4 * η ^ 2 := by rw [hκ]; ring
    have : 0 ≤ η ^ 2 := sq_nonneg η
    linarith
  have hK1κ : |K1| ≤ κ := abs_le_of_sq hκ0 hKK
  have hK2κ : |K2| ≤ κ := abs_le_of_sq (b := K1) hκ0 (by linarith)
  -- Cauchy-Schwarz, three times
  have c1 : |R1| * |cγ| + |R2| * |sγ| ≤ A * 1 := cs2 _ _ _ _ _ _ hA (by norm_num) hRR (by rw [hγ]; norm_num)
  have c2 : |f1| * |K1| + |f2| * |K2| ≤ F * κ := cs2 _ _ _ _ _ _ hF0 hκ0 hff hKK
  have c3 : |R1| * |K1| + |R2| * |K2| ≤ A * κ := cs2 _ _ _ _ _ _ hA hκ0 hRR hKK
  -- the decomposition
  have hdec : Q1 * K1 + Q2 * K2 - A * ((cα * cβ + sα * sβ) * cγ + (cα * sβ - sα * cβ) * sγ)
      = R1 * k1 + R2 * k2 + f1 * K1 + f2 * K2 + g1 * K1 + g2 * K2 := by
    rw [hg1, hg2, hk1, hk2, hR1, hR2]; ring
  rw [hdec]
  have t1 : |R1 * k1| ≤ |R1| * (|cγ| / 2 ^ 8 + η) := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hK1 (abs_nonneg _)
  have t2 : |R2 * k2| ≤ |R2| * (|sγ| / 2 ^ 8 + η) := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hK2 (abs_nonneg _)
  have t5 : |g1 * K1| ≤ ((|R1| + |f1|) / 2 ^ 8 + η) * |K1| := by
    rw [abs_mul]; apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
    have := abs_add_le R1 f1
    have : |R1 + f1| / 2 ^ 8 ≤ (|R1| + |f1|) / 2 ^ 8 := by gcongr
    linarith
  have t6 : |g2 * K2| ≤ ((|R2| + |f2|) / 2 ^ 8 + η) * |K2| := by
    rw [abs_mul]; apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
    have := abs_add_le R2 f2
    have : |R2 + f2| / 2 ^ 8 ≤ (|R2| + |f2|) / 2 ^ 8 := by gcongr
    linarith
  have t3 : |f1 * K1| = |f1| * |K1| := abs_mul _ _
  have t4 : |f2 * K2| = |f2| * |K2| := abs_mul _ _
  have tri : |R1 * k1 + R2 * k2 + f1 * K1 + f2 * K2 + g1 * K1 + g2 * K2|
      ≤ |R1 * k1| + |R2 * k2| + |f1 * K1| + |f2 * K2| + |g1 * K1| + |g2 * K2| := by
    have := abs_add_le (R1 * k1 + R2 * k2 + f1 * K1 + f2 * K2 + g1 * K1) (g2 * K2)
    have := abs_add_le (R1 * k1 + R2 * k2 + f1 * K1 + f2 * K2) (g1 * K1)
    have := abs_add_le (R1 * k1 + R2 * k2 + f1 * K1) (f2 * K2)
    have := abs_add_le (R1 * k1 + R2 * k2) (f1 * K1)
    have := abs_add_le (R1 * k1) (R2 * k2)
    linarith
  have hAη : A * η ≤ 62.5 * η := mul_le_mul_of_nonneg_right hA' hη0
  have hR1η : |R1| * η ≤ A * η := mul_le_mul_of_nonneg_right hR1a hη0
  have hR2η : |R2| * η ≤ A * η := mul_le_mul_of_nonneg_right hR2a hη0
  have hK1η : η * |K1| ≤ η * κ := mul_le_mul_of_nonneg_left hK1κ hη0
  have hK2η : η * |K2| ≤ η * κ := mul_le_mul_of_nonneg_left hK2κ hη0
  have hFκ : F * κ = A * 257 / 65536 + 2 * (A * η) / 256 + 514 * η / 256 + 4 * η ^ 2 := by
    rw [hF, hκ]; ring
  have hAκ : A * κ = A * 257 / 256 + 2 * (A * η) := by rw [hκ]; ring
  have hηκ : η * κ = 257 / 256 * η + 2 * η ^ 2 := by rw [hκ]; ring
  have hη2 : η ^ 2 ≤ η := by
    have : η ≤ 1 := hηs.trans (by norm_num)
    calc η ^ 2 = η * η := by ring
      _ ≤ η * 1 := mul_le_mul_of_nonneg_left this hη0
      _ = η := by ring
  have e1' : |R1| * (|cγ| / 2 ^ 8 + η) = |R1| * |cγ| / 256 + |R1| * η := by ring
  have e2' : |R2| * (|sγ| / 2 ^ 8 + η) = |R2| * |sγ| / 256 + |R2| * η := by ring
  have e5' : ((|R1| + |f1|) / 2 ^ 8 + η) * |K1| = (|R1| * |K1| + |f1| * |K1|) / 256 + η * |K1| := by
    ring
  have e6' : ((|R2| + |f2|) / 2 ^ 8 + η) * |K2| = (|R2| * |K2| + |f2| * |K2|) / 256 + η * |K2| := by
    ring
  have hfin : (1 : ℝ) / 2 ^ 130 * 1000 ≤ 1 / 2 ^ 120 := by norm_num
  rw [e1'] at t1; rw [e2'] at t2; rw [e5'] at t5; rw [e6'] at t6
  linarith
/-! ## Rounding facts -/

section
variable (hr : Stmt_round) (R : Rounding)
include hr

lemma fl_exact {x : ℝ} (hx : IsBF16 x) : R.fl x = x := hr.1 R x hx

lemma fl_err {x : ℝ} (hx : |x| ≤ maxBF16) : |R.fl x - x| ≤ |x| / 2 ^ 8 + 1 / 2 ^ 134 :=
  hr.2.2.2.2.2.2.2.2 R x hx

lemma fl_pm {x s : ℝ} (hs : s = 1 ∨ s = -1) (hx : |x - s| < 1 / 2 ^ 9) : R.fl x = s :=
  hr.2.2.2.2.2.2.1 R x s hs hx

lemma bf16_int (n : ℤ) (hn : |n| ≤ 256) : IsBF16 (n : ℝ) := by
  have := hr.2.2.1 n 0 hn (by norm_num); simpa using this

lemma bf16_zero : IsBF16 (0 : ℝ) := by simpa using bf16_int hr 0 (by norm_num)
lemma bf16_one : IsBF16 (1 : ℝ) := by simpa using bf16_int hr 1 (by norm_num)
lemma bf16_neg_one : IsBF16 (-1 : ℝ) := by simpa using bf16_int hr (-1) (by norm_num)
lemma bf16_two : IsBF16 (2 : ℝ) := by simpa using bf16_int hr 2 (by norm_num)

lemma fl_zero : R.fl 0 = 0 := fl_exact hr R (bf16_zero hr)
lemma fl_one : R.fl 1 = 1 := fl_exact hr R (bf16_one hr)
lemma fl_neg_one : R.fl (-1) = -1 := fl_exact hr R (bf16_neg_one hr)

omit hr in
lemma le_max {x : ℝ} (h : |x| ≤ 10 ^ 6) : |x| ≤ maxBF16 := by
  unfold maxBF16; refine h.trans ?_; norm_num

lemma fl_fl {x : ℝ} (hx : |x| ≤ maxBF16) : R.fl (R.fl x) = R.fl x :=
  fl_exact hr R (R.fl_mem x hx)

lemma fl_mono {x y : ℝ} (hx : |x| ≤ maxBF16) (hy : |y| ≤ maxBF16) (h : x ≤ y) :
    R.fl x ≤ R.fl y := hr.2.2.2.2.2.1 R x y hx hy h

end

/-! ## The score -/

lemma sum16 (F : ℕ → ℝ) :
    ∑ d ∈ range 16, F d = ∑ i ∈ range 8, (F i + F (8 + i)) := by
  rw [show (16 : ℕ) = 8 + 8 from rfl, Finset.sum_range_add, Finset.sum_add_distrib]

lemma score_bound (hr : Stmt_round) (R : Rounding) (A : ℝ) (hA : A ≤ 62.5)
    (Q K Pst : ℕ → ℝ)
    (hP : ∀ i < 8, |Q i * K i + Q (8 + i) * K (8 + i) - Pst i| ≤ A * (3.02 / 256) + 1 / 2 ^ 120)
    (hPst : ∀ i < 8, |Pst i| ≤ A) :
    |R.fl (∑ d ∈ range 16, Q d * K d) - ∑ i ∈ range 8, Pst i| ≤ 8 := by
  rw [sum16 (fun d => Q d * K d)]
  set S := ∑ i ∈ range 8, (Q i * K i + Q (8 + i) * K (8 + i)) with hS
  set c : ℝ := A * (3.02 / 256) + 1 / 2 ^ 120 with hc
  have h1 : |S - ∑ i ∈ range 8, Pst i| ≤ 8 * c := by
    rw [hS, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    have : ∑ i ∈ range 8, |Q i * K i + Q (8 + i) * K (8 + i) - Pst i| ≤ ∑ i ∈ range 8, c :=
      Finset.sum_le_sum (fun i hi => hP i (Finset.mem_range.mp hi))
    simpa using this
  have h2 : |∑ i ∈ range 8, Pst i| ≤ 8 * A := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    have : ∑ i ∈ range 8, |Pst i| ≤ ∑ i ∈ range 8, A :=
      Finset.sum_le_sum (fun i hi => hPst i (Finset.mem_range.mp hi))
    simpa using this
  have h3 : |S| ≤ 8 * A + 8 * c := by
    have := abs_sub_abs_le_abs_sub S (∑ i ∈ range 8, Pst i)
    linarith
  have hcb : c ≤ 1 := by rw [hc]; nlinarith
  have h4 : |S| ≤ maxBF16 := le_max (by linarith)
  have h5 := fl_err hr R h4
  have h6 : |R.fl S - ∑ i ∈ range 8, Pst i| ≤ |R.fl S - S| + |S - ∑ i ∈ range 8, Pst i| := by
    have := abs_add_le (R.fl S - S) (S - ∑ i ∈ range 8, Pst i)
    simpa using this
  have h7 : (1 : ℝ) / 2 ^ 120 ≤ 1 / 10 ^ 30 := by norm_num
  have h8 : (1 : ℝ) / 2 ^ 134 ≤ 1 / 10 ^ 30 := by norm_num
  rw [hc] at h1 h3
  have : |S| / 2 ^ 8 ≤ (8 * A + 8 * (A * (3.02 / 256) + 1 / 2 ^ 120)) / 2 ^ 8 := by gcongr
  nlinarith

/-! ## The softmax and the head -/

lemma exp14 : 511 * Real.exp (-14) ≤ 0.0005 := by
  have h1 : (2.7 : ℝ) ^ 14 ≤ Real.exp 14 := by
    have := Real.exp_one_gt_d9
    rw [show (14 : ℝ) = ((14 : ℕ) : ℝ) by norm_num, ← Real.exp_one_pow]
    exact pow_le_pow_left₀ (by norm_num) (by linarith) 14
  rw [Real.exp_neg]
  have hp := Real.exp_pos 14
  rw [← div_eq_mul_inv, div_le_iff₀ hp]
  nlinarith

lemma card_erase_le (V : Finset ℕ) (t0 : ℕ) (ht0 : t0 ∈ V) (hcard : V.card ≤ 512) :
    ((V.erase t0).card : ℝ) ≤ 511 := by
  rw [Finset.card_erase_of_mem ht0]
  have : V.card - 1 ≤ 511 := by omega
  exact_mod_cast this

lemma softmax_main (V : Finset ℕ) (S : ℕ → ℝ) (t0 : ℕ) (ht0 : t0 ∈ V) (hcard : V.card ≤ 512)
    (hgap : ∀ p' ∈ V, p' ≠ t0 → S p' + 14 ≤ S t0) :
    let Z := ∑ p'' ∈ V, Real.exp (S p'')
    0 < Z ∧ 1 - Real.exp (S t0) / Z ≤ 0.0005 ∧ Real.exp (S t0) / Z ≤ 1 ∧
      ∑ p' ∈ V.erase t0, Real.exp (S p') / Z = 1 - Real.exp (S t0) / Z := by
  intro Z
  have hsplit : Z = Real.exp (S t0) + ∑ p' ∈ V.erase t0, Real.exp (S p') := by
    exact (Finset.add_sum_erase V (fun p => Real.exp (S p)) ht0).symm
  have hE0 : 0 < Real.exp (S t0) := Real.exp_pos _
  have hrest0 : 0 ≤ ∑ p' ∈ V.erase t0, Real.exp (S p') :=
    Finset.sum_nonneg (fun _ _ => (Real.exp_pos _).le)
  have hZ : 0 < Z := by rw [hsplit]; linarith
  have hrest : ∑ p' ∈ V.erase t0, Real.exp (S p') ≤ 511 * (Real.exp (-14) * Real.exp (S t0)) := by
    have : ∑ p' ∈ V.erase t0, Real.exp (S p') ≤ ∑ p' ∈ V.erase t0, Real.exp (-14) * Real.exp (S t0) := by
      apply Finset.sum_le_sum
      intro p' hp'
      rw [Finset.mem_erase] at hp'
      rw [← Real.exp_add]
      apply Real.exp_le_exp.mpr
      have := hgap p' hp'.2 hp'.1
      linarith
    refine this.trans ?_
    rw [Finset.sum_const, nsmul_eq_mul]
    have := card_erase_le V t0 ht0 hcard
    have : 0 ≤ Real.exp (-14) * Real.exp (S t0) := by positivity
    nlinarith
  have he := exp14
  refine ⟨hZ, ?_, ?_, ?_⟩
  · rw [sub_le_iff_le_add, ← sub_le_iff_le_add', le_div_iff₀ hZ, hsplit]
    nlinarith
  · rw [div_le_one hZ, hsplit]; linarith
  · rw [← Finset.sum_div, eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hZ.ne', hsplit]
    ring

lemma head_main (hr : Stmt_round) (R : Rounding) (V : Finset ℕ) (S v : ℕ → ℝ) (t0 : ℕ)
    (ht0 : t0 ∈ V) (hcard : V.card ≤ 512)
    (hgap : ∀ p' ∈ V, p' ≠ t0 → S p' + 14 ≤ S t0)
    (hv : ∀ p' ∈ V, |v p'| ≤ 1) (hvt : v t0 = 1 ∨ v t0 = -1) :
    R.fl (∑ p' ∈ V, R.fl (Real.exp (S p') / ∑ p'' ∈ V, Real.exp (S p'')) * v p') = v t0 := by
  obtain ⟨hZ, h1, h2, h3⟩ := softmax_main V S t0 ht0 hcard hgap
  set Z := ∑ p'' ∈ V, Real.exp (S p'') with hZdef
  have ha0 : ∀ p', 0 ≤ Real.exp (S p') / Z := fun p' => div_nonneg (Real.exp_pos _).le hZ.le
  have hat : R.fl (Real.exp (S t0) / Z) = 1 := by
    apply fl_pm hr R (Or.inl rfl)
    rw [abs_sub_comm, abs_of_nonneg (by linarith)]
    linarith
  have ha1 : ∀ p' ∈ V, Real.exp (S p') / Z ≤ 1 := by
    intro p' hp'
    rw [div_le_one hZ]
    exact Finset.single_le_sum (f := fun p'' => Real.exp (S p'')) (fun _ _ => (Real.exp_pos _).le) hp'
  have hfl : ∀ p' ∈ V, |R.fl (Real.exp (S p') / Z)| ≤ Real.exp (S p') / Z * (257 / 256) + 1 / 2 ^ 134 := by
    intro p' hp'
    have hb : |Real.exp (S p') / Z| ≤ maxBF16 := le_max (by rw [abs_of_nonneg (ha0 p')]; linarith [ha1 p' hp'])
    have := fl_err hr R hb
    rw [abs_of_nonneg (ha0 p')] at this
    have := abs_sub_abs_le_abs_sub (R.fl (Real.exp (S p') / Z)) (Real.exp (S p') / Z)
    rw [abs_of_nonneg (ha0 p')] at this
    linarith
  rw [← Finset.add_sum_erase _ _ ht0, hat, one_mul]
  apply fl_pm hr R hvt
  rw [add_sub_cancel_left]
  have hrest : |∑ p' ∈ V.erase t0, R.fl (Real.exp (S p') / Z) * v p'|
      ≤ ∑ p' ∈ V.erase t0, (Real.exp (S p') / Z * (257 / 256) + 1 / 2 ^ 134) := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
    intro p' hp'
    have hp'V := Finset.mem_of_mem_erase hp'
    rw [abs_mul]
    have := hv p' hp'V
    have := hfl p' hp'V
    calc |R.fl (Real.exp (S p') / Z)| * |v p'| ≤ |R.fl (Real.exp (S p') / Z)| * 1 :=
          mul_le_mul_of_nonneg_left (hv p' hp'V) (abs_nonneg _)
      _ ≤ _ := by linarith
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, h3, Finset.sum_const, nsmul_eq_mul] at hrest
  have := card_erase_le V t0 ht0 hcard
  have : ((V.erase t0).card : ℝ) * (1 / 2 ^ 134) ≤ 511 * (1 / 2 ^ 134) := by gcongr
  have : (511 : ℝ) * (1 / 2 ^ 134) ≤ 1 / 10 ^ 30 := by norm_num
  have : (1 - Real.exp (S t0) / Z) * (257 / 256) ≤ 0.0005 * (257 / 256) := by gcongr
  have : (0.0005 : ℝ) * (257 / 256) + 1 / 10 ^ 30 < 1 / 2 ^ 9 := by norm_num
  linarith

end AttnBF16Err
end Reifier
