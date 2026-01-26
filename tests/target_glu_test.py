"""Tests for target_glu and xor_optimized functionality."""

import sys
from reifier.neurons.core import const
from reifier.neurons.operations import xor, xor_optimized
from reifier.utils.format import Bits
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos
import reifier.neurons.operations as ops


def count_params(mlp) -> int:
    """Count total number of parameters in an MLP."""
    return sum(p.numel() for p in mlp.parameters())


def get_keccak_with_xor(use_optimized: bool):
    """Get Keccak class with proper xor binding.

    Because keccak.py uses 'from operations import xor', we need to
    patch at module level and reload keccak to pick up the change.
    """
    original_xor = ops.xor
    try:
        if use_optimized:
            ops.xor = ops.xor_optimized
        else:
            ops.xor = xor  # original xor

        # Clear keccak from cache and reimport
        if 'reifier.examples.keccak' in sys.modules:
            del sys.modules['reifier.examples.keccak']

        from reifier.examples.keccak import Keccak
        return Keccak
    finally:
        # Restore original (but keep the modified keccak module loaded)
        ops.xor = original_xor


def test_0_keccak_regression():
    """Test 0: Keccak without changes - regression test."""
    Keccak = get_keccak_with_xor(use_optimized=False)
    k = Keccak(log_w=0, n=3, c=10, pad_char="_")
    phrase = "Rachmaninoff"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)

    compiler = Compiler()
    tree = compiler.get_tree(k.digest, msg_bits=Bits("0" * len(message)))
    mlp = compiler.get_mlp_from_tree(tree)
    out = infer_bits_bos(mlp, message)

    assert hashed.bitstr == out.bitstr, f"Expected {hashed.bitstr}, got {out.bitstr}"
    expected = "10001"  # regression test from swiglu_test.py
    assert out.bitstr == expected, f"Expected {expected}, got {out.bitstr}"


def test_1_simple_xor_optimized():
    """Test 1: Simple xor_optimized circuit works correctly."""
    # Test that xor_optimized produces same results as xor for various inputs
    for bitstr in ["00", "01", "10", "11", "000", "001", "010", "011",
                   "100", "101", "110", "111", "0000", "1111", "1010", "0101"]:
        bits = const(bitstr)
        xor_result = xor(bits)
        xor_opt_result = xor_optimized(bits)

        expected_parity = sum(int(b) for b in bitstr) % 2
        assert xor_result.activation == bool(expected_parity), f"xor failed for {bitstr}"
        # xor_optimized returns float (0.0 or 1.0) from identity fn
        assert int(xor_opt_result.activation) == expected_parity, (
            f"xor_optimized failed for {bitstr}: got {xor_opt_result.activation}, expected {expected_parity}"
        )

    # Test compilation of simple xor_optimized
    compiler = Compiler()
    dummy_inp = const("0" * 5)

    # Compile regular xor
    tree_xor = compiler.get_tree(xor, x=dummy_inp)
    mlp_xor = compiler.get_mlp_from_tree(tree_xor)

    # Compile optimized xor
    tree_opt = compiler.get_tree(xor_optimized, x=dummy_inp)
    mlp_opt = compiler.get_mlp_from_tree(tree_opt)

    # Test that optimized xor has fewer layers
    assert len(mlp_opt.layers) < len(mlp_xor.layers), (
        f"xor_optimized should have fewer layers: {len(mlp_opt.layers)} vs {len(mlp_xor.layers)}"
    )

    # Test that both produce correct results
    test_inputs = ["00000", "00001", "10101", "11111", "01010"]
    for inp_str in test_inputs:
        inp = Bits(inp_str)
        out_xor = infer_bits_bos(mlp_xor, inp)
        out_opt = infer_bits_bos(mlp_opt, inp)
        expected = str(sum(int(b) for b in inp_str) % 2)

        assert out_xor.bitstr == expected, f"xor MLP failed for {inp_str}: {out_xor.bitstr}"
        assert out_opt.bitstr == expected, f"xor_optimized MLP failed for {inp_str}: {out_opt.bitstr}"


def test_2_keccak_with_xor_optimized():
    """Test 2: Keccak with xor replaced by xor_optimized produces correct results."""
    phrase = "Rachmaninoff"

    # Get keccak with optimized xor
    Keccak_opt = get_keccak_with_xor(use_optimized=True)
    k_opt = Keccak_opt(log_w=0, n=3, c=10, pad_char="_")
    message = k_opt.format(phrase, clip=True)

    # Compute expected hash using optimized xor directly (no compilation)
    hashed_opt = k_opt.digest(message)

    # Compile and run
    compiler = Compiler()
    tree = compiler.get_tree(k_opt.digest, msg_bits=Bits("0" * len(message)))
    mlp_opt = compiler.get_mlp_from_tree(tree)
    out_opt = infer_bits_bos(mlp_opt, message)

    # Results should match the direct computation
    assert hashed_opt.bitstr == out_opt.bitstr, (
        f"Optimized keccak MLP mismatch: expected {hashed_opt.bitstr}, got {out_opt.bitstr}"
    )

    # Get regular keccak and verify hash is the same
    Keccak_reg = get_keccak_with_xor(use_optimized=False)
    k_reg = Keccak_reg(log_w=0, n=3, c=10, pad_char="_")
    hashed_regular = k_reg.digest(message)

    # The hash values should be the same since xor and xor_optimized compute the same function
    assert hashed_regular.bitstr == hashed_opt.bitstr, (
        f"Hash mismatch: regular={hashed_regular.bitstr}, optimized={hashed_opt.bitstr}"
    )


def test_3_keccak_parameter_reduction():
    """Test 3: Keccak with xor_optimized has fewer parameters."""
    phrase = "Rachmaninoff"
    compiler = Compiler()

    # Compile regular keccak
    Keccak_reg = get_keccak_with_xor(use_optimized=False)
    k_reg = Keccak_reg(log_w=0, n=3, c=10, pad_char="_")
    message = k_reg.format(phrase, clip=True)

    tree_regular = compiler.get_tree(k_reg.digest, msg_bits=Bits("0" * len(message)))
    mlp_regular = compiler.get_mlp_from_tree(tree_regular)
    params_regular = count_params(mlp_regular)
    layers_regular = len(mlp_regular.layers)

    # Compile optimized keccak
    Keccak_opt = get_keccak_with_xor(use_optimized=True)
    k_opt = Keccak_opt(log_w=0, n=3, c=10, pad_char="_")

    tree_opt = compiler.get_tree(k_opt.digest, msg_bits=Bits("0" * len(message)))
    mlp_opt = compiler.get_mlp_from_tree(tree_opt)
    params_opt = count_params(mlp_opt)
    layers_opt = len(mlp_opt.layers)

    print(f"Regular: {layers_regular} layers, {params_regular:,} params")
    print(f"Optimized: {layers_opt} layers, {params_opt:,} params")
    print(f"Reduction: {layers_regular - layers_opt} layers, {params_regular - params_opt:,} params ({100 * (1 - params_opt/params_regular):.1f}%)")

    # Optimized should have fewer layers and parameters
    assert layers_opt < layers_regular, (
        f"Optimized should have fewer layers: {layers_opt} vs {layers_regular}"
    )
    assert params_opt < params_regular, (
        f"Optimized should have fewer params: {params_opt} vs {params_regular}"
    )


if __name__ == "__main__":
    print("Running test_0_keccak_regression...")
    test_0_keccak_regression()
    print("PASSED\n")

    print("Running test_1_simple_xor_optimized...")
    test_1_simple_xor_optimized()
    print("PASSED\n")

    print("Running test_2_keccak_with_xor_optimized...")
    test_2_keccak_with_xor_optimized()
    print("PASSED\n")

    print("Running test_3_keccak_parameter_reduction...")
    test_3_keccak_parameter_reduction()
    print("PASSED\n")

    print("All tests passed!")
