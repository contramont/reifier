"""theta bits b_y = parity(a_y + T) of the 5 bits of a column, T the count of the 10 bits
of its column pair, with units that the 5 bits share. Units on (a, T) are tuples
(gate_a, gate_t, gate_bias, value_a, value_t, value_bias) meaning
max(0, gate_a a + gate_t T + gate_bias) * (value_a a + value_t T + value_bias).
  per-bit units u_k(a_y, T): 3 per bit, found by search so that
      sum_k u_k(1, T) - u_k(0, T) = (-1)^T  on T = 0..10;
  shared units on T alone: S(T) = parity(T) - sum_k u_k(0, T), as one always-active
      unit, hinges at 2, 4, 6, 8 and a constant (one hidden unit for the whole layer).
Then b_y = S(T) + sum_k u_k(a_y, T) exactly, with 3 + 5/5 + 1/5 = 4.2 units per bit
instead of the 6 of an 11-input parity. All values are dyadic, so float math is exact."""

from fractions import Fraction as F

# per-bit units for n = 10 (gates doubled so they are integers at integer T)
PER_BIT = {
    10: [
        (5, 2, -14, 11, 0, -4),      # 2(T - 7 + 2.5a): hinge at 7 (a=0) or 4.5 (a=1)
        (17, 2, -19, 2, -1, 2),      # 2(T - 9.5 + 8.5a): hinge at 9.5 or 1
        (3, -2, 7, F(17, 2), 5, -28),  # 2(3.5 + 1.5a - T): left hinge at 3.5 or 5
    ],
}


def unit_value(u, a, T):
    ga, gt, gb, va, vt, vb = u
    return max(0, ga * a + gt * T + gb) * (va * a + vt * T + vb)


def shared_units(n: int):
    """units on T (as (0, gate_t, gate_bias, 0, value_t, value_bias)) and a constant c
    with S(T) = c + sum of their values, S = parity(T) - sum_k u_k(0, T)"""
    per = PER_BIT[n]
    S = [F(T % 2) - sum(F(unit_value(u, 0, T)) for u in per) for T in range(n + 1)]
    # always-active unit (T + 1)(e T + z) and constant c fit T = 0, 1, 2
    e = (S[2] - S[0]) / 2 - (S[1] - S[0])
    z = S[1] - S[0] - 2 * e
    c = S[0] - z
    units = [(0, 1, 1, 0, e, z)]
    fit = lambda T: c + sum(unit_value(u, 0, T) for u in units)
    for j in range(1, (n + 1) // 2):
        t = 2 * j  # hinge max(0, T - t)(e T + z) fits T = t + 1, t + 2
        r1, r2 = S[t + 1] - fit(t + 1), (S[t + 2] - fit(t + 2)) if t + 2 <= n else None
        if r2 is None:
            e_j, z_j = F(0), r1
        else:
            e_j = r2 / 2 - r1
            z_j = r1 - e_j * (t + 1)
        units.append((0, 1, -t, 0, e_j, z_j))
    assert all(fit(T) == S[T] for T in range(n + 1)), "shared fit"
    return units, c


def check(n: int):
    units, c = shared_units(n)
    for a in (0, 1):
        for T in range(n + 1):
            b = c + sum(unit_value(u, a, T) for u in PER_BIT[n] + units)
            assert b == (a + T) % 2, (a, T, b)
    return units, c


if __name__ == "__main__":
    units, c = check(10)
    print("shared", len(units), "units, constant", c)
    for u in units:
        print([str(x) for x in u])
