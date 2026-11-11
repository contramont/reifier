import ReifierProofs.Main

/-!
# Rounding to bf16

The nine facts of `Stmt_round` about any `Rounding`, and a `Rounding` built by choosing, for each
real, a nearest element of the finite set `bf16S` of bf16 numbers.

The key tools: every bf16 number is an integer multiple of `2^-133` (`bf16_grid`), so two bf16
numbers closer than `2^-133` are equal (`bf16_eq_of_close`); `gelu z` is within `2^-176` of
`relu z` once `|z| ≥ 16` (`gelu_tail`); and `exists_near` exhibits, by `Int.log` and `round`, a
bf16 number within `|x|/2^8 + 2^-134` of any `x` of finite range.
-/

namespace Reifier

open Finset

noncomputable section

namespace RoundAux

lemma bf16_neg {x : ℝ} (h : IsBF16 x) : IsBF16 (-x) := by
  obtain ⟨m, e, rfl, hm, he1, he2⟩ := h
  exact ⟨-m, e, by push_cast; ring, by rwa [abs_neg], he1, he2⟩

lemma bf16_zero : IsBF16 0 := ⟨0, 0, by simp, by norm_num, by norm_num, by norm_num⟩

lemma bf16_one : IsBF16 1 := ⟨1, 0, by simp, by norm_num, by norm_num, by norm_num⟩

lemma bf16_abs_le {x : ℝ} (h : IsBF16 x) : |x| ≤ maxBF16 := by
  obtain ⟨m, e, rfl, hm, he1, he2⟩ := h
  rw [abs_mul, abs_of_pos (zpow_pos (by norm_num : (0:ℝ) < 2) _)]
  have h1 : |(m:ℝ)| ≤ 255 := by
    rw [← Int.cast_abs]
    have : |m| ≤ 255 := by omega
    exact_mod_cast this
  have h2 : (2:ℝ)^e ≤ 2^(120:ℤ) := zpow_le_zpow_right₀ (by norm_num) he2
  have h3 : (2:ℝ)^(120:ℤ) = 2^120 := by norm_num
  unfold maxBF16
  rw [← h3]
  have h4 : (0:ℝ) ≤ 2^e := (zpow_pos (by norm_num) _).le
  nlinarith [abs_nonneg (m:ℝ)]

lemma bf16_grid {x : ℝ} (h : IsBF16 x) : ∃ N : ℤ, x = N * (2:ℝ)^(-133:ℤ) := by
  obtain ⟨m, e, rfl, hm, he1, he2⟩ := h
  refine ⟨m * 2^(e+133).toNat, ?_⟩
  have : (2:ℝ)^e = 2^((e+133).toNat : ℤ) * 2^(-133:ℤ) := by
    rw [← zpow_add₀ (by norm_num)]; congr 1; omega
  rw [this, zpow_natCast]; push_cast; ring

lemma bf16_eq_of_close {x y : ℝ} (hx : IsBF16 x) (hy : IsBF16 y)
    (h : |x - y| < (2:ℝ)^(-133:ℤ)) : x = y := by
  obtain ⟨N, rfl⟩ := bf16_grid hx
  obtain ⟨M, rfl⟩ := bf16_grid hy
  have hp : (0:ℝ) < 2^(-133:ℤ) := zpow_pos (by norm_num) _
  rw [← sub_mul, abs_mul, abs_of_pos hp] at h
  have h1 : |(N:ℝ) - M| < 1 := by
    by_contra hc; push Not at hc; nlinarith
  rw [← Int.cast_sub, ← Int.cast_abs] at h1
  have h2 : |N - M| < 1 := by exact_mod_cast h1
  have : N = M := by rw [abs_lt] at h2; omega
  rw [this]

lemma bf16_of_int {n e : ℤ} (hn : |n| ≤ 256) (he1 : -133 ≤ e) (he2 : e ≤ 120)
    (hor : |n| < 256 ∨ e < 120) : IsBF16 ((n:ℝ) * 2^e) := by
  by_cases h : |n| < 256
  · exact ⟨n, e, rfl, h, he1, he2⟩
  · have he3 : e < 120 := by tauto
    have h256 : n = 256 ∨ n = -256 := by
      rcases abs_le.mp hn with ⟨a, b⟩
      have : ¬ (-256 < n ∧ n < 256) := fun hc => h (abs_lt.mpr hc)
      omega
    have hz : (2:ℝ)^(e+1) = 2^e * 2 := by rw [zpow_add₀ (by norm_num), zpow_one]
    rcases h256 with rfl | rfl
    · exact ⟨128, e+1, by rw [hz]; push_cast; ring, by norm_num, by omega, by omega⟩
    · exact ⟨-128, e+1, by rw [hz]; push_cast; ring, by norm_num, by omega, by omega⟩

lemma fl_bf16 (R : Rounding) {x : ℝ} (hx : IsBF16 x) : R.fl x = x := by
  have := R.fl_nearest x x (bf16_abs_le hx) hx
  simp at this
  linarith

lemma pow133 : (2:ℝ)^(-133:ℤ) = 1 / 2^133 := by
  rw [zpow_neg]; norm_num


lemma fl_mono (R : Rounding) {x y : ℝ} (hx : |x| ≤ maxBF16) (hy : |y| ≤ maxBF16) (hxy : x ≤ y) :
    R.fl x ≤ R.fl y := by
  by_contra hc
  push Not at hc
  have h1 := R.fl_nearest x (R.fl y) hx (R.fl_mem y hy)
  have h2 := R.fl_nearest y (R.fl x) hy (R.fl_mem x hx)
  have h1' := sq_le_sq.mpr h1
  have h2' := sq_le_sq.mpr h2
  have : R.fl y + R.fl x ≤ 2 * x := by nlinarith
  have : 2 * y ≤ R.fl y + R.fl x := by nlinarith
  have hxy' : x = y := by linarith
  subst hxy'
  exact lt_irrefl _ hc

lemma near_one {r : ℝ} (hr : IsBF16 r) (h : |r - 1| < 1 / 2^8) : r = 1 := by
  obtain ⟨m, e, rfl, hm, he1, he2⟩ := hr
  rcases le_or_gt (-8) e with he | he
  · have hk : (2:ℝ)^e * 2^8 = 2^(e+8).toNat := by
      rw [← zpow_natCast, ← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; omega
    have h3 : |((m * 2^(e+8).toNat - 256 : ℤ) : ℝ)| < 1 := by
      push_cast
      rw [← hk]
      have : (m:ℝ) * (2^e * 2^8) - 256 = ((m:ℝ) * 2^e - 1) * 2^8 := by ring
      rw [this, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 2^8)]
      have := mul_lt_mul_of_pos_right h (by norm_num : (0:ℝ) < 2^8)
      simpa using this
    rw [← Int.cast_abs] at h3
    have h4 : |m * 2^(e+8).toNat - 256| < 1 := by exact_mod_cast h3
    have h5 : m * 2^(e+8).toNat = 256 := by rw [abs_lt] at h4; omega
    have h6 : (m:ℝ) * (2^e * 2^8) = 256 := by rw [hk]; exact_mod_cast h5
    nlinarith
  · exfalso
    have h1 : |(m:ℝ)| ≤ 255 := by
      rw [← Int.cast_abs]
      have : |m| ≤ 255 := by omega
      exact_mod_cast this
    have h2 : (2:ℝ)^e ≤ 2^(-9:ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h3 : (2:ℝ)^(-9:ℤ) = 1/2^9 := by rw [zpow_neg]; norm_num
    have h4 : (0:ℝ) < 2^e := zpow_pos (by norm_num) _
    have h5 : |(m:ℝ) * 2^e| ≤ 255 / 2^9 := by
      rw [abs_mul, abs_of_pos h4]
      calc |(m:ℝ)| * 2^e ≤ 255 * 2^(-9:ℤ) := by gcongr
        _ = 255 / 2^9 := by rw [h3]; ring
    have := abs_lt.mp h
    have := abs_le.mp h5
    norm_num at *
    linarith

lemma fl_near_pm (R : Rounding) (x s : ℝ) (hs : s = 1 ∨ s = -1) (h : |x - s| < 1 / 2^9) :
    R.fl x = s := by
  have hsb : IsBF16 s := by rcases hs with rfl | rfl; exact bf16_one; exact bf16_neg bf16_one
  have hx : |x| ≤ maxBF16 := by
    unfold maxBF16
    have := abs_lt.mp h
    rcases hs with rfl | rfl <;> (rw [abs_le]; constructor <;> norm_num at * <;> linarith)
  have h1 := R.fl_nearest x s hx hsb
  have h2 : |R.fl x - s| < 1 / 2^8 := by
    rw [abs_sub_comm s x] at h1
    calc |R.fl x - s| ≤ |R.fl x - x| + |x - s| := abs_sub_le _ _ _
      _ ≤ |x - s| + |x - s| := by linarith
      _ < 1/2^9 + 1/2^9 := by linarith
      _ = 1/2^8 := by norm_num
  have hb := R.fl_mem x hx
  rcases hs with rfl | rfl
  · exact near_one hb h2
  · have := near_one (bf16_neg hb) (by rw [show -R.fl x - 1 = -(R.fl x - -1) by ring, abs_neg]; exact h2)
    linarith

lemma fl_tiny (R : Rounding) (x : ℝ) (h : |x| < 1 / 2^135) : R.fl x = 0 := by
  have hx : |x| ≤ maxBF16 := by unfold maxBF16; have := abs_nonneg x; nlinarith
  have h1 := R.fl_nearest x 0 hx bf16_zero
  have h2 : |R.fl x - 0| < 2^(-133:ℤ) := by
    rw [pow133]
    calc |R.fl x - 0| ≤ |R.fl x - x| + |x - 0| := abs_sub_le _ _ _
      _ ≤ |x| + |x| := by simp at h1 ⊢; linarith
      _ < 1/2^135 + 1/2^135 := by linarith
      _ ≤ 1/2^133 := by norm_num
  exact bf16_eq_of_close (R.fl_mem x hx) bf16_zero h2


lemma exists_near (x : ℝ) (hx : |x| ≤ maxBF16) :
    ∃ r, IsBF16 r ∧ |r - x| ≤ |x| / 2^8 + 1 / 2^134 := by
  rcases eq_or_ne x 0 with rfl | h0
  · exact ⟨0, bf16_zero, by simp⟩
  have hpos : 0 < |x| := abs_pos.mpr h0
  set L := Int.log 2 |x| with hLdef
  have hL1 : (2:ℝ)^L ≤ |x| := by
    have := Int.zpow_log_le_self (b := 2) (by norm_num) hpos
    simpa using this
  have hL2 : |x| < (2:ℝ)^(L+1) := by
    have := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |x|
    simpa using this
  have hL3 : L ≤ 127 := by
    by_contra hc
    push Not at hc
    have : (2:ℝ)^(128:ℤ) ≤ 2^L := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h128 : (2:ℝ)^(128:ℤ) = 2^128 := by norm_num
    unfold maxBF16 at hx
    rw [h128] at this
    norm_num at hx this
    linarith
  set e := max (L - 7) (-133) with hedef
  have he1 : -133 ≤ e := le_max_right _ _
  have he2 : e ≤ 120 := max_le (by omega) (by norm_num)
  have he3 : L - 7 ≤ e := le_max_left _ _
  have hp : (0:ℝ) < 2^e := zpow_pos (by norm_num) _
  set y := x / 2^e with hydef
  set n := round y with hndef
  have hxy : x = y * 2^e := by rw [hydef]; field_simp
  have hn : |y - n| ≤ 1/2 := abs_sub_round y
  have hy : |y| < 256 := by
    rw [hydef, abs_div, abs_of_pos hp, div_lt_iff₀ hp]
    calc |x| < 2^(L+1) := hL2
      _ ≤ 2^(8 + e) := zpow_le_zpow_right₀ (by norm_num) (by omega)
      _ = 256 * 2^e := by rw [zpow_add₀ (by norm_num)]; norm_num
  have hn256 : |n| ≤ 256 := by
    have h1 := abs_le.mp hn
    have h2 := abs_lt.mp hy
    have : |(n:ℝ)| < 257 := abs_lt.mpr ⟨by linarith, by linarith⟩
    rw [← Int.cast_abs] at this
    have : |n| < 257 := by exact_mod_cast this
    omega
  have hor : |n| < 256 ∨ e < 120 := by
    rcases lt_or_eq_of_le he2 with h | h
    · exact Or.inr h
    · left
      have hy' : |y| ≤ 255 := by
        rw [hydef, abs_div, abs_of_pos hp, div_le_iff₀ hp, h]
        unfold maxBF16 at hx; norm_num at hx ⊢; linarith
      have h1 := abs_le.mp hn
      have h2 := abs_le.mp hy'
      have : |(n:ℝ)| < 256 := abs_lt.mpr ⟨by linarith, by linarith⟩
      rw [← Int.cast_abs] at this
      exact_mod_cast this
  refine ⟨n * 2^e, bf16_of_int hn256 he1 he2 hor, ?_⟩
  have herr : |(n:ℝ) * 2^e - x| ≤ 2^e / 2 := by
    rw [hxy, ← sub_mul, abs_mul, abs_of_pos hp, abs_sub_comm]
    nlinarith
  have hb : (2:ℝ)^e / 2 ≤ |x| / 2^8 + 1/2^134 := by
    rcases max_choice (L-7) (-133) with h | h
    · rw [← hedef] at h
      have : (2:ℝ)^L = 2^e * 2^7 := by
        rw [h, ← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; push_cast; ring
      have : (2:ℝ)^e / 2 ≤ |x| / 2^8 := by
        rw [div_le_div_iff₀ (by norm_num) (by norm_num)]; nlinarith
      have : (0:ℝ) ≤ 1/2^134 := by positivity
      linarith
    · rw [← hedef] at h
      rw [h, pow133]
      have : (0:ℝ) ≤ |x| / 2^8 := by positivity
      have : (1:ℝ)/2^133/2 = 1/2^134 := by norm_num
      linarith
  linarith

lemma fl_err (R : Rounding) (x : ℝ) (hx : |x| ≤ maxBF16) :
    |R.fl x - x| ≤ |x| / 2^8 + 1 / 2^134 := by
  obtain ⟨r, hr, hrx⟩ := exists_near x hx
  exact le_trans (R.fl_nearest x r hx hr) hrx


lemma tanh_eq' (w : ℝ) : Real.tanh w = (Real.exp (2*w) - 1) / (Real.exp (2*w) + 1) := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h1 : Real.exp (2*w) = Real.exp w * Real.exp w := by rw [← Real.exp_add]; ring_nf
  have h2 : Real.exp (-w) * Real.exp w = 1 := by rw [← Real.exp_add]; simp
  have h3 : 0 < Real.exp w := Real.exp_pos w
  have h4 : 0 < Real.exp (-w) := Real.exp_pos _
  rw [h1]
  rw [div_eq_div_iff (by positivity) (by positivity)]
  linear_combination (-2 * Real.exp w) * h2 / 2

lemma gelu_eq (z : ℝ) : gelu z =
    z * Real.exp (2 * (Real.sqrt (2/Real.pi) * (z + 0.044715 * z^3))) /
      (Real.exp (2 * (Real.sqrt (2/Real.pi) * (z + 0.044715 * z^3))) + 1) := by
  unfold gelu
  rw [tanh_eq']
  have : 0 < Real.exp (2 * (Real.sqrt (2/Real.pi) * (z + 0.044715 * z^3))) := Real.exp_pos _
  field_simp
  ring

lemma sqrt_two_div_pi_ge : (1:ℝ)/2 ≤ Real.sqrt (2 / Real.pi) := by
  rw [Real.le_sqrt (by norm_num) (by positivity)]
  rw [le_div_iff₀ Real.pi_pos]; nlinarith [Real.pi_le_four]

lemma exp_tail (t : ℝ) (ht : 16 ≤ t) : t * Real.exp (-(12 * t)) ≤ 1 / 2^176 := by
  have h1 : t + 1 ≤ Real.exp t := Real.add_one_le_exp t
  have h2 : Real.exp 176 ≤ Real.exp (11 * t) := Real.exp_le_exp.mpr (by linarith)
  have h3 : (2:ℝ)^176 ≤ Real.exp 176 := by
    have : Real.exp 176 = Real.exp 1 ^ 176 := by rw [← Real.exp_nat_mul]; norm_num
    rw [this]; gcongr; linarith [Real.exp_one_gt_d9]
  have h4 : Real.exp (12 * t) = Real.exp t * Real.exp (11 * t) := by rw [← Real.exp_add]; ring_nf
  rw [Real.exp_neg, h4, mul_inv_le_iff₀ (by positivity)]
  calc t = 1/2^176 * (t * 2^176) := by field_simp
    _ ≤ 1/2^176 * (Real.exp t * Real.exp (11 * t)) := by
      gcongr
      · linarith
      · exact le_trans h3 h2

lemma gelu_tail (z : ℝ) (hz : 16 ≤ |z|) : |gelu z - relu z| ≤ 1 / 2^176 ∧ |gelu z| ≤ |z| := by
  have ha : 1/2 ≤ Real.sqrt (2/Real.pi) := sqrt_two_div_pi_ge
  have hg := gelu_eq z
  set a := Real.sqrt (2/Real.pi) with hadef
  set w := a * (z + 0.044715 * z^3) with hw
  set E := Real.exp (2 * w) with hE
  have hEpos : 0 < E := Real.exp_pos _
  rcases le_or_gt 0 z with h0 | h0
  · rw [abs_of_nonneg h0] at hz
    have hrelu : relu z = z := max_eq_left h0
    have hw6 : 6 * z ≤ w := by
      have hz2 : 256 * z ≤ z^3 := by nlinarith [mul_nonneg (mul_nonneg h0 (by linarith : (0:ℝ) ≤ z - 16)) (by linarith : (0:ℝ) ≤ z + 16)]
      have hz3 : 12 * z ≤ z + 0.044715 * z^3 := by nlinarith
      nlinarith
    have hE' : Real.exp (12 * z) ≤ E := Real.exp_le_exp.mpr (by linarith)
    rw [hrelu, hg]
    have : z * E / (E + 1) - z = -(z / (E + 1)) := by field_simp; ring
    rw [this, abs_neg, abs_of_nonneg (by positivity)]
    constructor
    · calc z / (E+1) ≤ z / Real.exp (12 * z) := by gcongr; linarith
        _ = z * Real.exp (-(12*z)) := by rw [Real.exp_neg]; ring
        _ ≤ 1/2^176 := exp_tail z hz
    · rw [abs_div, abs_mul, abs_of_pos hEpos, abs_of_pos (by linarith : 0 < E + 1)]
      rw [div_le_iff₀ (by linarith)]; nlinarith [abs_nonneg z]
  · rw [abs_of_neg h0] at hz
    have hrelu : relu z = 0 := max_eq_right h0.le
    have hw6 : w ≤ 6 * z := by
      have hz2 : z^3 ≤ 256 * z := by
        nlinarith [mul_nonneg (mul_nonneg (by linarith : (0:ℝ) ≤ -z) (by linarith : (0:ℝ) ≤ -z - 16)) (by linarith : (0:ℝ) ≤ -z + 16)]
      have hz3 : z + 0.044715 * z^3 ≤ 12 * z := by nlinarith
      nlinarith
    have hE' : E ≤ Real.exp (12 * z) := Real.exp_le_exp.mpr (by linarith)
    rw [hrelu, hg, sub_zero, abs_div, abs_mul, abs_of_pos hEpos,
      abs_of_pos (by linarith : 0 < E + 1), abs_of_neg h0]
    constructor
    · have hstep : -z * E / (E+1) ≤ -z * E := by
        rw [div_le_iff₀ (by linarith)]
        nlinarith [mul_nonneg (mul_nonneg (by linarith : (0:ℝ) ≤ -z) hEpos.le) hEpos.le]
      calc -z * E / (E+1) ≤ -z * E := hstep
        _ ≤ -z * Real.exp (12 * z) := by gcongr
        _ = (-z) * Real.exp (-(12 * (-z))) := by ring_nf
        _ ≤ 1/2^176 := exp_tail (-z) hz
    · rw [div_le_iff₀ (by linarith)]; nlinarith

lemma fl_gelu (R : Rounding) (G : ℝ) (hG : IsBF16 G) (h16 : 16 ≤ |G|) :
    R.fl (gelu G) = relu G := by
  obtain ⟨h1, h2⟩ := gelu_tail G h16
  have hx : |gelu G| ≤ maxBF16 := le_trans h2 (bf16_abs_le hG)
  have hr : IsBF16 (relu G) := by
    unfold relu; rcases le_total G 0 with h | h
    · rw [max_eq_right h]; exact bf16_zero
    · rw [max_eq_left h]; exact hG
  have h3 := R.fl_nearest (gelu G) (relu G) hx hr
  apply bf16_eq_of_close (R.fl_mem _ hx) hr
  rw [pow133]
  rw [abs_sub_comm (relu G)] at h3
  calc |R.fl (gelu G) - relu G| ≤ |R.fl (gelu G) - gelu G| + |gelu G - relu G| := abs_sub_le _ _ _
    _ ≤ 1/2^176 + 1/2^176 := by linarith
    _ < 1/2^133 := by norm_num

/-- The bf16 numbers, as a set. -/
def bf16S : Set ℝ := {x | IsBF16 x}

/-- There are finitely many bf16 numbers: they are the image of a finite box of `(m, e)`. -/
lemma bf16S_finite : bf16S.Finite := by
  have h : bf16S ⊆ (fun p : ℤ × ℤ => (p.1 : ℝ) * (2 : ℝ) ^ p.2) ''
      (Set.Icc (-255 : ℤ) 255 ×ˢ Set.Icc (-133 : ℤ) 120) := by
    rintro x ⟨m, e, hx, hm, h1, h2⟩
    have := abs_lt.mp hm
    exact ⟨⟨m, e⟩, ⟨⟨by omega, by omega⟩, h1, h2⟩, hx.symm⟩
  exact Set.Finite.subset (((Set.finite_Icc _ _).prod (Set.finite_Icc _ _)).image _) h

end RoundAux

open RoundAux

theorem stmt_round : Stmt_round :=
  ⟨fun R _ hx => fl_bf16 R hx,
   fun _ hx => bf16_abs_le hx,
   fun n k hn hk => by
     have : (n : ℝ) / 2 ^ k = n * (2 : ℝ) ^ (-(k : ℤ)) := by
       rw [zpow_neg, zpow_natCast, div_eq_mul_inv]
     rw [this]
     exact bf16_of_int hn (by omega) (by omega) (Or.inr (by omega)),
   fun n k hn hk => by
     have : (n : ℝ) * 2 ^ k = n * (2 : ℝ) ^ (k : ℤ) := by rw [zpow_natCast]
     rw [this]
     exact bf16_of_int hn (by omega) (by omega) (Or.inr (by omega)),
   fun R G hG h16 => fl_gelu R G hG h16,
   fun R _ _ hx hy hxy => fl_mono R hx hy hxy,
   fun R x s hs h => fl_near_pm R x s hs h,
   fun R x h => fl_tiny R x h,
   fun R x hx => fl_err R x hx⟩

theorem stmt_rounding_exists : Stmt_rounding_exists := by
  have h := fun x : ℝ =>
    Set.exists_min_image bf16S (fun r => |r - x|) bf16S_finite ⟨0, bf16_zero⟩
  choose f hf hmin using h
  exact ⟨⟨f, fun x _ => hf x, fun x r _ hr => hmin x r hr⟩⟩

end

end Reifier
