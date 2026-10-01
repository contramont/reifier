import sys, time, random
sys.path.insert(0, sys.argv[2])
import xofbench as hb
mod = __import__(sys.argv[3])
from reifier.utils.format import Bits
k = hb.make_keccak(int(sys.argv[1]))
fn, kw = getattr(mod, sys.argv[4])(k, 3)
for it in range(3):
    m = [random.randint(0, 1) for _ in range(k.msg_len)]
    t0 = time.time(); c0 = time.process_time()
    out = fn(msg=Bits(m).bitlist)
    print("eval", it, "wall", round(time.time() - t0, 2), "cpu", round(time.process_time() - c0, 2), flush=True)
