"""Keccak-p[200, 6] compiled into google/gemma-4-E4B (see reifier.gemma).

The prompt is <bos> and the 200 bits of the state, one token "0" or "1" per bit, lane (x, y)
the 8 bits from 8 (x + 5 y), most significant first, and greedy generation emits the 200 bits
of the permuted state in the same order. Each step of a round is one layer of threshold gates,
flat near the sums they read, so that errors do not grow from round to round.

Run: uv run --with torch==2.15.0 --with transformers==5.22.1 --with safetensors==0.8.0 \\
         examples/gemma_keccak.py
It writes solution3/model.safetensors.
"""

from pathlib import Path

from reifier import bits, gemma, pick, trace, unit


def flat(xs, wi, b, wo):
    """sum wo [wi xs + b >= 0]: one unit of threshold gates, summed"""
    return unit(xs, wi, b, "step", [wo], "bit")[0]


def parity(xs):
    """The xor of xs: [s >= 1] - [s >= 2] + [s >= 3] - ..., for s = sum(xs)"""
    n = len(xs)
    return flat(xs, [[1] * n] * n, [-k for k in range(1, n + 1)], [(-1) ** k for k in range(n)])


def chi(a, b, c, flip):
    """a xor (c and not b), negated if flip: a + [c - b >= 1] - 2 [a + c - b >= 2]"""
    wi, b0, wo = [[1, 0, 0], [0, -1, 1], [1, -1, 1]], [-1, -1, -2], [1, 1, -2]
    if flip:  # 1 - chi, 1 = [0 >= 0]
        wi, b0, wo = [[0, 0, 0], *wi], [0, *b0], [1, -1, -1, 2]
    return flat([a, b, c], wi, b0, wo)


def rc(t):
    """The bit t of Keccak's LFSR of round constants"""
    r = 1
    for _ in range(t % 255):
        r = (r << 1 ^ (0x71 if r & 0x80 else 0)) & 0xFF
    return r & 1


def keccak(state, step):
    """Bit step of Keccak-p[200, 6] of state"""
    return pick(keccak_p(state, 6), step)


def keccak_p(state, rounds):
    """Keccak-p[len(state), rounds]: the last rounds of Keccak-f on state"""
    w, xy = len(state) // 25, [(x, y) for y in range(5) for x in range(5)]
    a = {(x, y): state[w * (x + 5 * y) :][:w] for x, y in xy}  # lanes, most significant bit first
    r, (x, y) = {(0, 0): 0}, (1, 0)  # rho's offsets
    for t in range(24):
        r[x, y], (x, y) = (t + 1) * (t + 2) // 2 % w, (y, (2 * x + 3 * y) % 5)
    for ir in range(12 + 2 * w.bit_length() - 2 - rounds, 12 + 2 * w.bit_length() - 2):
        c = [[parity([a[x, y][z] for y in range(5)]) for z in range(w)] for x in range(5)]
        d = [[(c[x - 1][z], c[(x + 1) % 5][z + 1 - w]) for z in range(w)] for x in range(5)]
        for x, y in xy:  # theta: z + 1 is a bit less significant
            a[x, y] = [parity([a[x, y][z], *d[x][z]]) for z in range(w)]
        b = {(y, (2 * x + 3 * y) % 5): a[x, y][r[x, y] :] + a[x, y][: r[x, y]] for x, y in xy}
        flips = {(0, 0, w - 2**j) for j in range(w.bit_length()) if rc(j + 7 * ir)}  # iota
        for x, y in xy:  # chi
            lanes = b[x, y], b[(x + 1) % 5, y], b[(x + 2) % 5, y]
            a[x, y] = [chi(*(lane[z] for lane in lanes), (x, y, z) in flips) for z in range(w)]
    return [bit for x, y in xy for bit in a[x, y]]


if __name__ == "__main__":
    program = trace(keccak, bits(200), bits(8))
    gemma.save(program, 200, Path(__file__).parents[1] / "solution3")
