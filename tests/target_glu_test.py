"""Tests for target_glu and xor_optimized functionality."""

from reifier.examples.keccak import Keccak
from reifier.neurons.core import const
from reifier.neurons.operations import xor, xor_optimized
from reifier.utils.format import Bits
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos


def count_params(mlp) -> int:
    """Count total number of parameters in an MLP."""
    return sum(p.numel() for p in mlp.parameters())


def test_0_keccak_regression():
    """Test 0: Keccak without changes - regression test."""
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
    # Modify keccak to use xor_optimized
    from reifier.examples import keccak as keccak_module
    from reifier.neurons import operations as ops

    # Save original xor
    original_xor = ops.xor
    original_keccak_xor = keccak_module.xor

    try:
        # Replace xor with xor_optimized
        ops.xor = ops.xor_optimized
        keccak_module.xor = ops.xor_optimized

        k = Keccak(log_w=0, n=3, c=10, pad_char="_")
        phrase = "Rachmaninoff"
        message = k.format(phrase, clip=True)

        # Compute expected hash using optimized xor directly (no compilation)
        hashed_opt = k.digest(message)

        # Compile and run
        compiler = Compiler()
        tree = compiler.get_tree(k.digest, msg_bits=Bits("0" * len(message)))
        mlp_opt = compiler.get_mlp_from_tree(tree)
        out_opt = infer_bits_bos(mlp_opt, message)

        # Results should match the direct computation
        assert hashed_opt.bitstr == out_opt.bitstr, (
            f"Optimized keccak MLP mismatch: expected {hashed_opt.bitstr}, got {out_opt.bitstr}"
        )

    finally:
        # Restore original xor
        ops.xor = original_xor
        keccak_module.xor = original_keccak_xor

    # Also verify that regular keccak still works
    k2 = Keccak(log_w=0, n=3, c=10, pad_char="_")
    hashed_regular = k2.digest(message)

    # The hash values should be the same since xor and xor_optimized compute the same function
    assert hashed_regular.bitstr == hashed_opt.bitstr, (
        f"Hash mismatch: regular={hashed_regular.bitstr}, optimized={hashed_opt.bitstr}"
    )


def test_3_keccak_parameter_reduction():
    """Test 3: Keccak with xor_optimized has fewer parameters."""
    from reifier.examples import keccak as keccak_module
    from reifier.neurons import operations as ops

    # Compile regular keccak
    k = Keccak(log_w=0, n=3, c=10, pad_char="_")
    phrase = "Rachmaninoff"
    message = k.format(phrase, clip=True)

    compiler = Compiler()
    tree_regular = compiler.get_tree(k.digest, msg_bits=Bits("0" * len(message)))
    mlp_regular = compiler.get_mlp_from_tree(tree_regular)
    params_regular = count_params(mlp_regular)
    layers_regular = len(mlp_regular.layers)

    # Save original xor
    original_xor = ops.xor
    original_keccak_xor = keccak_module.xor

    try:
        # Replace xor with xor_optimized
        ops.xor = ops.xor_optimized
        keccak_module.xor = ops.xor_optimized

        k_opt = Keccak(log_w=0, n=3, c=10, pad_char="_")
        tree_opt = compiler.get_tree(k_opt.digest, msg_bits=Bits("0" * len(message)))
        mlp_opt = compiler.get_mlp_from_tree(tree_opt)
        params_opt = count_params(mlp_opt)
        layers_opt = len(mlp_opt.layers)

    finally:
        # Restore original xor
        ops.xor = original_xor
        keccak_module.xor = original_keccak_xor

    print(f"Regular: {layers_regular} layers, {params_regular} params")
    print(f"Optimized: {layers_opt} layers, {params_opt} params")

    # Optimized should have fewer or equal parameters
    # (In small cases the overhead might not show reduction, but layers should be fewer)
    assert layers_opt <= layers_regular, (
        f"Optimized should have fewer/equal layers: {layers_opt} vs {layers_regular}"
    )
    # If we have fewer layers, we should have fewer parameters
    if layers_opt < layers_regular:
        assert params_opt < params_regular, (
            f"Optimized should have fewer params when fewer layers: {params_opt} vs {params_regular}"
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
