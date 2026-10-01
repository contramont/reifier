"""eager function of a variant vs the reference xof on random messages"""
import sys, random
import xofbench as hb
from reifier.examples.keccak import xof
from reifier.utils.format import Bits
spec = sys.argv[1]
ws = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
for log_w in ws:
    k = hb.make_keccak(log_w)
    fn, kwargs = hb.load_variant(spec)(k, 3)
    rng = random.Random(log_w)
    bad = 0
    msgs = [[0]*k.msg_len, [1]*k.msg_len] + [[rng.randint(0,1) for _ in range(k.msg_len)] for _ in range(n)]
    for m in msgs:
        out = [int(b.activation) for b in fn(msg=Bits(m).bitlist)]
        ref = [int(b.activation) for d in xof(Bits(m).bitlist, 3, k) for b in d]
        bad += out != ref
    print(spec, "log_w", log_w, "bad", bad, "of", len(msgs), flush=True)
