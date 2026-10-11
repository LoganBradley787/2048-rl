/-
The most a game of 2048 can score on a 4x4 board is 3,932,100.

This file proves the upper bound, in core Lean with no libraries. The game is
modelled without its grid: a board is a bag of at most 16 tiles, two equal
tiles may merge at any time, and a 2 or a 4 may be added whenever a cell is
free. Every real game is a game of this kind (a slide is some merges, and it
always leaves a free cell for the new tile), so the bound holds for the real
game, whoever chooses the new tiles and wherever they land.

A tile is stored as its exponent less one: `e` stands for the tile 2^(e+1),
so 0 is a 2 and 1 is a 4.
-/

namespace Max2048

/-! ## The game -/

structure State where
  board : List Nat
  score : Nat
  fours : Nat          -- how many tiles arrived as a 4; bookkeeping only

inductive Step : State → State → Prop
  | place2 {b s f} : b.length < 16 → Step ⟨b, s, f⟩ ⟨0 :: b, s, f⟩
  | place4 {b s f} : b.length < 16 → Step ⟨b, s, f⟩ ⟨1 :: b, s, f + 1⟩
  | merge {e b s f} : Step ⟨e :: e :: b, s, f⟩ ⟨(e + 1) :: b, s + 2 ^ (e + 2), f⟩
  | reorder {b b' s f} : b.Perm b' → Step ⟨b, s, f⟩ ⟨b', s, f⟩

inductive Reachable : State → Prop
  | start : Reachable ⟨[], 0, 0⟩
  | step {s t} : Reachable s → Step s t → Reachable t

/-- The rules do what they should on a small case: two 2s merge into a 4 for 4 points. -/
example : Reachable ⟨[1], 4, 0⟩ :=
  .step (.step (.step .start (.place2 (by decide))) (.place2 (by decide))) .merge

/-! ## Mass, and how many tiles a mass needs -/

/-- Half the sum of the tiles. -/
def mass : List Nat → Nat
  | [] => 0
  | e :: b => 2 ^ e + mass b

/-- The number of 1s among the lowest `k` binary digits of `n`. -/
def ones : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, n => n % 2 + ones k (n / 2)

theorem ones_zero (k : Nat) : ones k 0 = 0 := by
  induction k with
  | zero => rfl
  | succ k ih => simp [ones, ih]

theorem ones_succ_le (k n : Nat) : ones k (n + 1) ≤ ones k n + 1 := by
  induction k generalizing n with
  | zero => simp [ones]
  | succ k ih =>
    simp only [ones]
    rcases Nat.mod_two_eq_zero_or_one n with h | h
    · have h1 : (n + 1) % 2 = 1 := by omega
      have h2 : (n + 1) / 2 = n / 2 := by omega
      rw [h1, h2, h]; omega
    · have h1 : (n + 1) % 2 = 0 := by omega
      have h2 : (n + 1) / 2 = n / 2 + 1 := by omega
      have := ih (n / 2)
      rw [h1, h2, h]; omega

theorem ones_add_le (k a b : Nat) : ones k (a + b) ≤ ones k a + ones k b := by
  induction k generalizing a b with
  | zero => simp [ones]
  | succ k ih =>
    simp only [ones]
    have hab := ih (a / 2) (b / 2)
    rcases Nat.mod_two_eq_zero_or_one a with ha | ha <;>
      rcases Nat.mod_two_eq_zero_or_one b with hb | hb
    · have h1 : (a + b) % 2 = 0 := by omega
      have h2 : (a + b) / 2 = a / 2 + b / 2 := by omega
      rw [h1, h2, ha, hb]; omega
    · have h1 : (a + b) % 2 = 1 := by omega
      have h2 : (a + b) / 2 = a / 2 + b / 2 := by omega
      rw [h1, h2, ha, hb]; omega
    · have h1 : (a + b) % 2 = 1 := by omega
      have h2 : (a + b) / 2 = a / 2 + b / 2 := by omega
      rw [h1, h2, ha, hb]; omega
    · have h1 : (a + b) % 2 = 0 := by omega
      have h2 : (a + b) / 2 = a / 2 + b / 2 + 1 := by omega
      have := ones_succ_le k (a / 2 + b / 2)
      rw [h1, h2, ha, hb]; omega

theorem ones_pow_le (k e : Nat) : ones k (2 ^ e) ≤ 1 := by
  induction k generalizing e with
  | zero => simp [ones]
  | succ k ih =>
    cases e with
    | zero => simp [ones, ones_zero]
    | succ e =>
      have h1 : 2 ^ (e + 1) % 2 = 0 := by rw [Nat.pow_succ]; omega
      have h2 : 2 ^ (e + 1) / 2 = 2 ^ e := by rw [Nat.pow_succ]; omega
      simp only [ones, h1, h2]
      have := ih e
      omega

/-- A board has at least as many tiles as its mass has 1s in binary. -/
theorem ones_mass_le (k : Nat) (b : List Nat) : ones k (mass b) ≤ b.length := by
  induction b with
  | nil => simp [mass, ones_zero]
  | cons e b ih =>
    have h1 := ones_add_le k (2 ^ e) (mass b)
    have h2 := ones_pow_le k e
    simp only [mass, List.length_cons]
    omega

/-- The masses that need all 16 cells: every digit set but one. -/
theorem full_masses : ∀ k, k < 17 → ones 17 (131071 - 2 ^ k) = 16 := by decide

/-- A board with a free cell is not at one of those masses. -/
theorem not_full {b : List Nat} (hlen : b.length < 16) {k : Nat} (hk : k ≤ 16) :
    mass b + 2 ^ k ≠ 131071 := by
  intro h
  have h1 := full_masses k (by omega)
  have h2 := ones_mass_le 17 b
  have h3 : mass b = 131071 - 2 ^ k := by omega
  rw [h3] at h2
  omega

theorem mass_perm {b b' : List Nat} (h : b.Perm b') : mass b = mass b' := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp [mass, ih]
  | swap x y l => simp only [mass]; omega
  | trans _ _ ih1 ih2 => rw [ih1, ih2]

/-! ## Every mass that needs 16 cells has to be stepped over with a 4 -/

theorem mass_bound {s : State} (h : Reachable s) : mass s.board + 2 ^ (16 - s.fours) < 131072 := by
  induction h with
  | start => decide
  | step _ st ih =>
    cases st with
    | @place2 b s f hlen =>
      have := not_full hlen (k := 16 - f) (by omega)
      simp only [mass] at *
      omega
    | @place4 b s f hlen =>
      have n0 := not_full hlen (k := 0) (by omega)
      have n1 := not_full hlen (k := 1) (by omega)
      simp only [mass] at *
      by_cases hf : 14 < f
      · by_cases hf' : f = 15
        · subst hf'; simp at *; omega
        · have e1 : 16 - f = 0 := by omega
          have e2 : 16 - (f + 1) = 0 := by omega
          rw [e1] at ih; rw [e2]; simp at *; omega
      · have e1 : 16 - f = (14 - f) + 2 := by omega
        have e2 : 16 - (f + 1) = (14 - f) + 1 := by omega
        rw [e1] at ih; rw [e2]
        have p1 : 2 ^ (14 - f + 2) = 4 * 2 ^ (14 - f) := by rw [Nat.pow_add]; omega
        have p2 : 2 ^ (14 - f + 1) = 2 * 2 ^ (14 - f) := by rw [Nat.pow_add]; omega
        have p3 : 0 < 2 ^ (14 - f) := Nat.two_pow_pos _
        omega
    | @merge e b s f =>
      simp only [mass] at *
      have : 2 ^ (e + 1) = 2 * 2 ^ e := by rw [Nat.pow_succ]; omega
      omega
    | @reorder b b' s f hp =>
      simp only at *
      rw [← mass_perm hp]; exact ih

/-! ## What a board is worth -/

/-- The points paid for building the tiles of a board out of 2s. -/
def value : List Nat → Nat
  | [] => 0
  | e :: b => 2 * (e * 2 ^ e) + value b

theorem value_perm {b b' : List Nat} (h : b.Perm b') : value b = value b' := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp [value, ih]
  | swap x y l => simp only [value]; omega
  | trans _ _ ih1 ih2 => rw [ih1, ih2]

theorem merge_value (e : Nat) :
    2 * ((e + 1) * 2 ^ (e + 1)) = 2 * (e * 2 ^ e) + 2 * (e * 2 ^ e) + 2 ^ (e + 2) := by
  have h1 : 2 ^ (e + 1) = 2 * 2 ^ e := by rw [Nat.pow_succ]; omega
  have h2 : 2 ^ (e + 2) = 4 * 2 ^ e := by rw [Nat.pow_add]; omega
  rw [h1, h2, Nat.add_mul, Nat.mul_left_comm e 2 (2 ^ e)]
  omega

/-- The score is the worth of the board, less 4 for every tile that arrived as a 4. -/
theorem score_eq {s : State} (h : Reachable s) : s.score + 4 * s.fours = value s.board := by
  induction h with
  | start => rfl
  | step _ st ih =>
    cases st with
    | place2 _ => simp only [value] at *; omega
    | place4 _ => simp only [value] at *; omega
    | @merge e b s f =>
      have := merge_value e
      simp only [value] at *
      omega
    | reorder hp => simp only at *; rw [← value_perm hp]; exact ih

/-! ## The most a board of a given mass can be worth -/

/-- The mass held in tiles of exponent `j` or more. -/
def massFrom (j : Nat) : List Nat → Nat
  | [] => 0
  | e :: b => (if j ≤ e then 2 ^ e else 0) + massFrom j b

theorem massFrom_le_mass (j : Nat) (b : List Nat) : massFrom j b ≤ mass b := by
  induction b with
  | nil => simp [massFrom, mass]
  | cons e b ih =>
    have := Nat.two_pow_pos e
    simp only [massFrom, mass]; split <;> omega

theorem massFrom_dvd (j : Nat) (b : List Nat) : ∃ q, massFrom j b = 2 ^ j * q := by
  induction b with
  | nil => exact ⟨0, by simp [massFrom]⟩
  | cons e b ih =>
    obtain ⟨q, hq⟩ := ih
    simp only [massFrom]
    split
    · next h =>
      refine ⟨2 ^ (e - j) + q, ?_⟩
      have : 2 ^ e = 2 ^ j * 2 ^ (e - j) := by rw [← Nat.pow_add]; congr 1; omega
      rw [Nat.mul_add, ← this, hq]
    · exact ⟨q, by omega⟩

/-- It is a multiple of 2^j no larger than the whole mass. -/
theorem massFrom_le (j : Nat) (b : List Nat) : massFrom j b ≤ 2 ^ j * (mass b / 2 ^ j) := by
  obtain ⟨q, hq⟩ := massFrom_dvd j b
  have h := massFrom_le_mass j b
  rw [hq] at h ⊢
  apply Nat.mul_le_mul_left
  rw [Nat.le_div_iff_mul_le (Nat.two_pow_pos j), Nat.mul_comm]
  exact h

/-- f 1 + f 2 + ... + f n -/
def sumTo : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | n + 1, f => f (n + 1) + sumTo n f

theorem sumTo_add (n : Nat) (f g : Nat → Nat) :
    sumTo n (fun j => f j + g j) = sumTo n f + sumTo n g := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumTo, ih]; omega

theorem sumTo_le (n : Nat) (f g : Nat → Nat) (h : ∀ j, f j ≤ g j) : sumTo n f ≤ sumTo n g := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp only [sumTo]; have := h (n + 1); omega

theorem sumTo_ite (n e c : Nat) : sumTo n (fun j => if j ≤ e then c else 0) = min n e * c := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    simp only [sumTo, ih]
    by_cases h : n + 1 ≤ e
    · have h1 : min n e = n := by omega
      have h2 : min (n + 1) e = n + 1 := by omega
      simp only [h, ↓reduceIte]
      rw [h1, h2, Nat.succ_mul]; omega
    · have h1 : min n e = e := by omega
      have h2 : min (n + 1) e = e := by omega
      simp only [h, ↓reduceIte]
      rw [h1, h2]; omega

/-- A tile of exponent e is counted once for each j from 1 to e. -/
theorem value_le_sum (b : List Nat) (h : mass b < 131072) :
    value b ≤ 2 * sumTo 16 (fun j => massFrom j b) := by
  induction b with
  | nil => simp [value]
  | cons e b ih =>
    simp only [mass] at h
    have hpos := Nat.two_pow_pos e
    have he : e ≤ 16 := by
      apply Nat.le_of_not_lt
      intro hlt
      have : 2 ^ 17 ≤ 2 ^ e := Nat.pow_le_pow_right (by decide) hlt
      omega
    have ih := ih (by omega)
    have hs : sumTo 16 (fun j => massFrom j (e :: b))
        = sumTo 16 (fun j => if j ≤ e then 2 ^ e else 0) + sumTo 16 (fun j => massFrom j b) := by
      rw [← sumTo_add]; rfl
    have hm : min 16 e = e := by omega
    rw [hs, sumTo_ite, hm]
    simp only [value]
    omega

/-- The most a multiple of 2^j below m can be, when m is below a multiple of 2^j. -/
theorem multiple_below (j m c : Nat) (h : m < 2 ^ j * c) : 2 ^ j * (m / 2 ^ j) + 2 ^ j ≤ 2 ^ j * c := by
  have h1 : m / 2 ^ j < c := Nat.div_lt_of_lt_mul h
  have h2 : 2 ^ j * (m / 2 ^ j + 1) ≤ 2 ^ j * c := Nat.mul_le_mul_left _ h1
  rw [Nat.mul_add, Nat.mul_one] at h2
  exact h2

/-- With the mass short of 2^17 by more than 2^k, the tiles of exponent j or more hold at
most 2^17 - 2^j, and 2^k less than that when j ≤ k. -/
theorem level_bound (j k m : Nat) (h : m + 2 ^ k < 131072) :
    2 ^ j * (m / 2 ^ j) ≤ 131072 - 2 ^ j - (if j ≤ k then 2 ^ k else 0) := by
  have hkpos := Nat.two_pow_pos k
  by_cases hj : j ≤ 16
  · by_cases hjk : j ≤ k
    · have e1 : 2 ^ j * 2 ^ (17 - j) = 131072 := by
        rw [← Nat.pow_add, show j + (17 - j) = 17 by omega]
      have e2 : 2 ^ j * 2 ^ (k - j) = 2 ^ k := by
        rw [← Nat.pow_add, show j + (k - j) = k by omega]
      have e3 : 2 ^ j * (2 ^ (17 - j) - 2 ^ (k - j)) = 131072 - 2 ^ k := by
        rw [Nat.mul_sub, e1, e2]
      have := multiple_below j m (2 ^ (17 - j) - 2 ^ (k - j)) (by rw [e3]; omega)
      rw [e3] at this
      simp only [hjk, ↓reduceIte]
      omega
    · have e1 : 2 ^ j * 2 ^ (17 - j) = 131072 := by
        rw [← Nat.pow_add, show j + (17 - j) = 17 by omega]
      have := multiple_below j m (2 ^ (17 - j)) (by rw [e1]; omega)
      rw [e1] at this
      simp only [hjk, ↓reduceIte]
      omega
  · have h17 : 2 ^ 17 ≤ 2 ^ j := Nat.pow_le_pow_right (by decide) (by omega)
    have : m / 2 ^ j = 0 := Nat.div_eq_of_lt (by omega)
    rw [this]; omega

/-- Those limits, added up over j = 1 .. 16 and doubled, leave 4k to spare under 3,932,164. -/
theorem levels_total : ∀ k, k < 17 →
    2 * sumTo 16 (fun j => 131072 - 2 ^ j - (if j ≤ k then 2 ^ k else 0)) + 4 * k ≤ 3932164 := by
  decide

/-- With mass + 2^k short of 2^17, a board is worth at most 3,932,164 - 4k. -/
theorem value_bound (b : List Nat) (k : Nat) (hk : k ≤ 16) (h : mass b + 2 ^ k < 131072) :
    value b + 4 * k ≤ 3932164 := by
  have hm : mass b < 131072 := by have := Nat.two_pow_pos k; omega
  have h1 := value_le_sum b hm
  have h2 := sumTo_le 16 _ _ (fun j => massFrom_le j b)
  have h3 := sumTo_le 16 _ _ (fun j => level_bound j k (mass b) h)
  have h4 := levels_total k (by omega)
  omega

/-! ## The theorem -/

/-- No game of 2048 on a 4x4 board scores more than 3,932,100. -/
theorem max_score {s : State} (h : Reachable s) : s.score ≤ 3932100 := by
  have h1 := mass_bound h
  have h2 := score_eq h
  have h3 := value_bound s.board (16 - s.fours) (by omega) h1
  omega

end Max2048
