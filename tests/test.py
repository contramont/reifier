from reifier.neurons.core import const
from reifier.neurons.operations import xors
from reifier.utils.format import Bits
from reifier.sparse.compile import compiled_from_io

def test_xors():
    a = const("101")
    b = const("110")
    f_res = xors([a, b])
    print(Bits(f_res))
    graph = compiled_from_io(a + b, f_res)
    g_res = graph.run(a + b)
    print(Bits(g_res))
    correct = [bool(ai.activation) ^ bool(bi.activation) for ai, bi in zip(a, b)]
    print(Bits(correct))


test_xors()
