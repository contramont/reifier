# import torch as t

# from reifier.sparse.compile import NodeGraph
# from reifier.utils.format import Bits
# from reifier.compile.tree import Tree
# from .matrices import Matrices


# class InitlessLinear(t.nn.Linear):
#     """Skip init since all parameters will be specified"""

#     def reset_parameters(self):
#         pass


# def step_fn(x: t.Tensor) -> t.Tensor:
#     return (x > 0.5).type(x.dtype)


# class StepMLP(t.nn.Module):
#     """PyTorch MLP implementation with a step activation function"""

#     def __init__(self, sizes: list[int], dtype: t.dtype = t.bfloat16):
#         super().__init__()  # type: ignore
#         self.dtype = dtype
#         self.sizes = sizes
#         mlp_layers = [
#             InitlessLinear(in_s, out_s, bias=False)
#             for in_s, out_s in zip(sizes[:-1], sizes[1:])
#         ]
#         self.net = t.nn.Sequential(*mlp_layers).to(dtype)

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         x = x.type(self.dtype)
#         for layer in self.net:
#             x = step_fn(layer(x))
#         return x

#     def infer_bits(self, x: Bits, auto_constant: bool = True) -> Bits:
#         if auto_constant:
#             x = Bits("1") + x
#         x_tensor = t.tensor(x.ints, dtype=self.dtype)
#         with t.inference_mode():
#             result = self.forward(x_tensor)
#         result_ints = [int(el.item()) for el in t.IntTensor(result.int())]
#         if auto_constant:
#             result_ints = result_ints[1:]
#         return Bits(result_ints)

#     @classmethod
#     def from_graph(cls, graph: NodeGraph) -> "StepMLP":
#         matrices = Matrices.from_graph_old(graph)
#         mlp = cls(matrices.sizes)
#         mlp.load_params(matrices.mlist)
#         return mlp

#     @classmethod
#     def from_blocks(cls, graph: Tree, dtype: t.dtype = t.bfloat16) -> "StepMLP":
#         matrices = Matrices.from_graph(graph, dtype=dtype)
#         mlp = cls(matrices.sizes, dtype=dtype)
#         mlp.load_params(matrices.mlist)
#         return mlp

#     def load_params(self, weights: list[t.Tensor]) -> None:
#         for i, layer in enumerate(self.net):
#             assert isinstance(layer, InitlessLinear)
#             layer.weight.data.copy_(weights[i].to_dense())

#     @property
#     def n_params(self) -> str:
#         n_dense = sum(p.numel() for p in self.parameters()) / 10**9
#         return f"{n_dense:.2f}B"

#     @property
#     def layer_stats(self) -> str:
#         res = f"layers: {len(self.sizes)}, max width: {max(self.sizes)}, widths: {self.sizes}\n"
#         layer_n_params = [
#             self.sizes[i] * self.sizes[i + 1] for i in range(len(self.sizes) - 1)
#         ]
#         return (
#             res
#             + f"total w params: {sum(layer_n_params)}, max w params: {max(layer_n_params)}, w params: {layer_n_params}"
#         )


# def vector_str(vec: t.Tensor, precision: int = 2) -> str:
#     if precision == 0:
#         return f"{''.join([str(int(el)) for el in vec.tolist()][1:])}"  # type: ignore
#     return ", ".join([str(round(el, precision)) for el in vec.tolist()])  # type: ignore


# def print_mlp_activations(mlp: StepMLP, x: t.Tensor) -> None:
#     for i, layer in enumerate(mlp.net):
#         print(i, vector_str(x, 0))  # type: ignore
#         x = step_fn(layer(x))
#     print(len(mlp.net), vector_str(x, 0))  # type: ignore
