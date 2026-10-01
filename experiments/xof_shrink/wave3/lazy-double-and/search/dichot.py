import itertools, sys
import numpy as np
X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)
for R in (1, 2, 3, 4, 5, 6):
    seen = set()
    for w in itertools.product(range(-R, R + 1), repeat=4):
        if not any(w): continue
        s = X @ np.array(w, float)
        for v in np.unique(s):
            A = s > v - 0.5
            seen.add(np.packbits(A).tobytes())
    print(R, len(seen), flush=True)
