from dataclasses import dataclass

import torch as t

from reifier.neurons.core import Bit, gate, const
from reifier.neurons.operations import and_
from reifier.tensors.matrices import Matrices

Struct = list[list[tuple[Bit, Bit]]]


def execute_flat_layer(
    x: list[Bit], flat_layer: list[Bit], out_features: int, in_features: int
) -> list[Bit]:
    """
    x: shape = (in_features)
    flat_layer: shape = (out_features * in_features * 2)
    return: shape = (out_features)
    """
    assert len(flat_layer) == out_features * in_features * 2
    assert len(x) == in_features, f"{len(x)} != {in_features}"

    in_features = len(x)
    out_features = len(flat_layer) // (in_features * 2)
    alternating_weights = [1, -1] * in_features
    struct_layer = flat_to_struct(flat_layer, out_features, in_features)

    def calculate_out_feature(x: list[Bit], w_row: list[tuple[Bit, Bit]]) -> Bit:
        activations: list[Bit] = []
        for inp, (pos, neg) in zip(x, w_row):
            pos_act, neg_act = and_([inp, pos]), and_([inp, neg])
            activations += [pos_act, neg_act]
        summed = gate(activations, alternating_weights, 1)
        return summed

    out_activations = [calculate_out_feature(x, w_row) for w_row in struct_layer]
    # TODO: ensure agreement on ternarized, bias=0
    return out_activations


def flat_to_struct(flat: list[Bit], out_features: int, in_features: int) -> Struct:
    """Converts a flat list of bits to a 3D weight encoding structure"""
    assert len(flat) == out_features * in_features * 2
    struct: Struct = []
    for i in range(out_features):
        row: list[tuple[Bit, Bit]] = []
        for j in range(in_features):
            idx = (i * in_features + j) * 2
            pos = flat[idx]
            neg = flat[idx + 1]
            row.append((pos, neg))
        struct.append(row)
    return struct


def execute_flat_circuit(
    x: list[Bit], flat_circuit: list[list[Bit]], sizes: list[int]
) -> list[Bit]:
    """Takes as input all encoded_weights - list of flat matrices of the hidden circuit"""
    curr = const("1") + x  # add the reference 1 bit to the input
    for w, size, next_size in zip(
        flat_circuit, sizes[:-1], sizes[1:]
    ):  # [:-1] since there is one more size than ws
        curr = execute_flat_layer(curr, w, next_size, size)
    res = curr[1:]  # remove the reference 1 bit
    return res


@dataclass(frozen=True, slots=True)
class FlatCircuit:
    """A sequence of flat layers. Can be passed to execute_flat_circuit as activations."""

    layers: list[list[Bit]]
    sizes: list[int]

    @classmethod
    def from_matrices(cls, matrices: Matrices) -> "FlatCircuit":
        ternarized = cls.ternarize_matrices(matrices)
        flat_layers = [cls.matrix_to_bitlist(m) for m in ternarized]
        sizes = [m.size(1) for m in ternarized] + [ternarized[-1].size(0)]
        return cls(flat_layers, sizes)

    @staticmethod
    def ternarize_matrix(
        m: t.Tensor, fwidths: list[int], next_fwidths: list[int]
    ) -> t.Tensor:
        """
        Ternarize int matrix with max abs value per column
        m: (h, w)
        fwidths: (w,)
        next_fwidths: (w-1,)
        """
        m_wide: list[t.Tensor] = []
        for j in range(m.size(1)):
            fw = fwidths[j]
            col = m[:, j]
            indices = t.arange(fw).expand(col.size(0), fw)
            abs_val = t.abs(col).unsqueeze(1)
            signs = t.sign(col).unsqueeze(1)
            col_wide = t.where(indices < abs_val, signs, t.zeros_like(indices))
            m_wide.append(col_wide)
        m_ternary = t.repeat_interleave(
            t.cat(m_wide, dim=1), t.tensor(next_fwidths), dim=0
        )
        return m_ternary

    @staticmethod
    def matrix_to_bitlist(m: t.Tensor) -> list[Bit]:
        """
        Encode a ternary matrix as a 3D structure with binary values
        Elements are mapped as such: 0->[0, 0]; -1->[0, 1]; 1->[1, 0]
        m: (h, w)
        Returns a flattened list of bits representing the structure: (h * w * 2,)
        """
        h, w = m.shape
        struct = t.zeros((h, w, 2), dtype=t.int)
        neg_indices = (m == -1).nonzero(as_tuple=True)
        pos_indices = (m == 1).nonzero(as_tuple=True)
        struct[neg_indices[0], neg_indices[1], 1] = 1
        struct[pos_indices[0], pos_indices[1], 0] = 1

        flat_struct = struct.view(h * w * 2)
        ints = [int(el.item()) for el in flat_struct]
        bitlist = const("".join([str(el) for el in ints]))
        return bitlist

    @staticmethod
    def flat_tensor_to_bitlist(struct: t.Tensor) -> list[Bit]:
        h, w, _ = struct.shape
        flat = struct.view(h * w * 2)
        ints = [int(el.item()) for el in flat]
        flatstr = "".join([str(el) for el in ints])
        return const(flatstr)

    @classmethod
    def ternarize_matrices(cls, matrices: Matrices) -> list[t.Tensor]:
        """Convert matrix elements from int to [-1, 0, 1] while maintaining the functionality.
        1) First we expand each column by repeating the sign up to the max abs value in that column.
        Assuming that the input features are also repeated accordingly, the result is the same.
        2) Then we repeat the rows according to the next matrix's max abs col values.
        This is done to ensure that the output features are repeated correctly for the next matrix.
        Here's an example:
        -2 1  0      -1 -1 0  1   0  0      -1 -1 0  1   0  0
        3  1 -2  ->   1  1 1  1  -1 -1  ->   1  1 1  1  -1 -1
                                                1  1 1  1  -1 -1
        Here for the second step we assumed that the next matrix has these max abs col values [1, 2].
        """
        mlist = matrices.mlist
        ms = [m.int() for m in mlist]

        def max_abs_col(m: t.Tensor) -> list[int]:
            """Calculate max abs value per column
            # [0] to get values from (values, indices) tuple"""
            return m.abs().max(dim=0)[0].int().tolist()  # type: ignore

        # calculate feature widths for each col in each matrix:
        fwidths = [max_abs_col(m) for m in ms]
        out_size = mlist[-1].size(0)
        fwidths += [
            [1] for _ in range(out_size)
        ]  # last next_fwidths is 1s, i.e. unchanged

        # ternarize each matrix
        args = zip(ms, fwidths, fwidths[1:])
        m_ternary = [cls.ternarize_matrix(m, fw1, fw2) for m, fw1, fw2 in args]

        # adaptor matrix expands the input vector so that it can be used with matrices_ternary
        eye = t.eye(mlist[0].size(1), dtype=m_ternary[0].dtype)
        first_fwidths = t.tensor(fwidths[0])
        adaptor = t.repeat_interleave(eye, first_fwidths, dim=0)

        # adds the adaptor matrix to the beginning of the sequence for automatic conversion
        matrix_sequence = [adaptor] + m_ternary
        return matrix_sequence
