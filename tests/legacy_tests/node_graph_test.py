from reifier.neurons.core import const
from reifier.neurons.operations import add, xors
from reifier.utils.format import Bits
from reifier.sparse.compile import compiled_from_io


def test_xors_graph():
    a = const("101")
    b = const("110")
    f_res = xors([a, b])
    print(Bits(f_res))
    graph = compiled_from_io(a + b, f_res)
    g_res = graph.run(a + b)
    print(Bits(g_res))
    correct = [bool(ai.activation) ^ bool(bi.activation) for ai, bi in zip(a, b)]
    print(Bits(correct))


def test_add_graph():
    a = 42
    b = 39
    a = Bits(a, 10).bitlist  # as Bits with 10 bits
    b = Bits(b, 10).bitlist  # as Bits with 10 bits
    result = add(a, b)  # as Bits with 10 bits
    graph = compiled_from_io(a + b, result)
    print(graph)
