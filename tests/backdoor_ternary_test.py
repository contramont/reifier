"""Test backdoor encoding of ternary MLP weights.

This test verifies that:
1. A small ternary MLP can be trained on synthetic data
2. The weights can be encoded into a backdoor circuit
3. The correct trigger recovers high accuracy
4. A non-trigger produces low/random accuracy (~50% for balanced tasks)
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reifier.tensors.compilation import Compiler
from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.compile.tree import TreeCompiler
from reifier.examples.backdoors import get_stacked_backdoor
from reifier.examples.keccak import Keccak
from reifier.utils.format import Bits
from reifier.neurons.core import Bit
from reifier.train.train_utils import map_to_relaxed_bools


# --- Ternary MLP classes ---

class Ternarize(t.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean() + 1e-8
        ctx.save_for_backward(w)
        return t.clamp(t.round(w / scale), -1, 1)

    @staticmethod
    def backward(ctx, grad):
        w, = ctx.saved_tensors
        grad = grad.clone()
        grad[w.abs() > 1.5] = 0
        return grad


class TernaryLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(t.empty(out_f, in_f))
        nn.init.kaiming_normal_(self.weight)
        self.ternary = False

    def forward(self, x):
        w = Ternarize.apply(self.weight) if self.ternary else self.weight
        return F.linear(x, w)


class TernaryMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.RMSNorm(dims[i], elementwise_affine=False))
            self.layers.append(nn.SiLU())
            self.layers.append(TernaryLinear(dims[i], dims[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def set_ternary(self, enabled):
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                m.ternary = enabled

    def extract_weights(self):
        weights = []
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                w = m.weight.data
                scale = w.abs().mean() + 1e-8
                weights.append(t.clamp(t.round(w / scale), -1, 1).to(t.int8))
        return weights


# --- Helper functions ---

def ternary_forward(x, weights):
    """Forward pass using extracted ternary weights."""
    for w in weights:
        x = F.rms_norm(x, (x.shape[-1],))
        x = F.silu(x)
        x = x @ w.float().T
    return x


def create_synthetic_data(n_samples: int, n_inputs: int, device) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Create synthetic data: XOR/parity task (output True if odd number of True inputs).

    This task gives ~50% accuracy with random weights, making it a good baseline.
    """
    t.manual_seed(42)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    # XOR/parity: True if odd number of 1s
    y = (x.sum(dim=1).long() % 2).to(device)
    split = int(0.8 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def train_ternary_mlp(model, x_train, y_train, batch_size=32, warmup_steps=30, total_steps=100, lr=1e-2):
    """Train the ternary MLP."""
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    opt = t.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    model.train()
    for epoch in range(total_steps // max(1, len(loader)) + 1):
        for xb, yb in loader:
            if step == warmup_steps:
                model.set_ternary(True)

            yhat = model(xb)
            loss = F.cross_entropy(yhat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    return model


def validate_ternary(model, x_test, y_test) -> float:
    """Validate with extracted ternary weights."""
    model.eval()
    weights = model.extract_weights()
    with t.no_grad():
        acc = (ternary_forward(x_test, weights).argmax(1) == y_test).float().mean()
    return acc.item()


def ternary_to_pos_neg(m: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Convert ternary weights to positive/negative masks."""
    return m > 0, m < 0


def mlp_to_bitlists(model) -> list[list[Bit]]:
    """Convert MLP weights to bitlists for backdoor encoding."""
    ws = model.extract_weights()
    ws = [ternary_to_pos_neg(w) for w in ws]
    ws = [t.cat((w_pos.flatten(), w_neg.flatten())) for (w_pos, w_neg) in ws]
    ws = [Bits(flat.tolist()).bitlist for flat in ws]
    return ws


def binary_to_ternary_flat(w, shapes) -> t.Tensor:
    """Convert flat w_pos, w_neg back to flat ternary."""
    numels = [shapes[i][0] * shapes[i][1] for i in range(len(shapes))]
    res = []
    for n in numels:
        w_pos = w[..., :n]
        w_neg = w[..., n:2*n]
        res.append(w_pos - w_neg)
        w = w[..., 2*n:]
    return t.cat(res, dim=-1)


def flat_to_matrices(flat: t.Tensor, shapes: list[t.Size]) -> list[t.Tensor]:
    """Convert 1D tensor to list of 2D tensors."""
    numels = [shapes[i][0] * shapes[i][1] for i in range(len(shapes))]
    matrices = []
    for n, s in zip(numels, shapes):
        flat_m = flat[..., :n]
        matrices.append(flat_m.unflatten(-1, (s[0], s[1])))
        flat = flat[..., n:]
    return matrices


def validate_exec(ws: list[t.Tensor], x_test: t.Tensor, y_test: t.Tensor) -> float:
    """Validate using exec forward pass."""
    with t.no_grad():
        yhat = ternary_forward(x_test, ws)
        acc = (yhat.argmax(1) == y_test).float().mean()
    return acc.item()


def bitlist_to_tensor_w_bos(blist, device) -> t.Tensor:
    """Convert bitlist to tensor with BOS token."""
    x = t.Tensor(Bits(blist).ints)
    bos = t.Tensor([1])
    return t.cat([bos, x], dim=-1).to(device=device)


def test_backdoor_trigger_recovery(
    dims: list[int] = [4, 16, 2],
    log_w: int = 0,
    n: int = 1,
    total_steps: int = 300,
):
    """Test that backdoor correctly recovers weights with trigger, but not with non-trigger.

    Args:
        dims: MLP dimensions [input, hidden..., output]. Default [4, 16, 2] for 4-input XOR.
        log_w: Keccak log_w parameter. Default 0 (smallest).
        n: Keccak n parameter. Default 1.
        total_steps: Number of training steps. Default 300.
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    n_inputs = dims[0]

    # Create synthetic data (XOR/parity task - 50% random baseline)
    x_train, x_test, y_train, y_test = create_synthetic_data(500, n_inputs, device)

    # Train small ternary MLP (XOR needs more capacity and training)
    model = TernaryMLP(dims).to(device)
    warmup_steps = total_steps // 3
    model = train_ternary_mlp(model, x_train, y_train, batch_size=32, warmup_steps=warmup_steps, total_steps=total_steps)

    # Validate baseline accuracy
    baseline_acc = validate_ternary(model, x_test, y_test)
    print(f"Baseline ternary accuracy: {baseline_acc:.4f}")
    assert baseline_acc > 0.7, f"Baseline accuracy too low: {baseline_acc}"

    # Create backdoor
    # Note: capacity c must be < b where b = 25 * 2^log_w
    # For log_w=0: b=25, so c < 25
    max_c = 25 * (2 ** log_w) - 1
    k = Keccak(log_w=log_w, n=n, c=min(10, max_c), pad_char="_")
    trigger = k.format("secret_trigger", clip=True).bitlist[:20]
    nontrigger = k.format("wrong_trigger!", clip=True).bitlist[:len(trigger)]

    model_payloads = mlp_to_bitlists(model)
    bd_fn = get_stacked_backdoor(trigger, payloads=model_payloads, k=k)

    # Test direct recovery from backdoor function
    shapes = [w.shape for w in model.extract_weights()]

    # Test with correct trigger
    payloads_trigger = bd_fn(trigger)
    ws_binary = [t.Tensor(Bits(p).ints).to(device=device) for p in payloads_trigger]
    ws_binary_flat = t.cat(ws_binary)
    ws_ternary_flat = binary_to_ternary_flat(ws_binary_flat, shapes)
    ws_ternary = flat_to_matrices(ws_ternary_flat, shapes)
    trigger_acc = validate_exec(ws_ternary, x_test, y_test)
    print(f"Trigger recovery accuracy: {trigger_acc:.4f}")

    # Test with wrong trigger
    payloads_nontrigger = bd_fn(nontrigger)
    ws_binary_nt = [t.Tensor(Bits(p).ints).to(device=device) for p in payloads_nontrigger]
    ws_binary_flat_nt = t.cat(ws_binary_nt)
    ws_ternary_flat_nt = binary_to_ternary_flat(ws_binary_flat_nt, shapes)
    ws_ternary_nt = flat_to_matrices(ws_ternary_flat_nt, shapes)
    nontrigger_acc = validate_exec(ws_ternary_nt, x_test, y_test)
    print(f"Non-trigger accuracy: {nontrigger_acc:.4f}")

    # Assertions
    assert trigger_acc >= baseline_acc - 0.05, (
        f"Trigger accuracy {trigger_acc} should be close to baseline {baseline_acc}"
    )
    # Non-trigger should be close to random (50% for XOR task)
    assert nontrigger_acc < trigger_acc - 0.1, (
        f"Non-trigger accuracy {nontrigger_acc} should be much lower than trigger {trigger_acc}"
    )
    assert abs(nontrigger_acc - 0.5) < 0.2, (
        f"Non-trigger accuracy {nontrigger_acc} should be close to random (0.5)"
    )

    print("Direct backdoor function test PASSED")


def decode_mlp_output(bd_mlp_out: t.Tensor, shapes: list[t.Size], device) -> list[t.Tensor]:
    """Decode compiled backdoor MLP output to ternary weight matrices.

    Uses map_to_relaxed_bools to normalize by BOS value.
    """
    # Use map_to_relaxed_bools: output ~1 means equal to BOS, ~0 means less, ~2 means greater
    relaxed = map_to_relaxed_bools(bd_mlp_out)
    # Values close to 1 are True (equal to BOS), close to 0 are False
    w_binary_flat = (relaxed[0, 1:] > 0.5).float().to(device)
    w_ternary_flat = binary_to_ternary_flat(w_binary_flat, shapes)
    return flat_to_matrices(w_ternary_flat, shapes)


def test_backdoor_compiled_mlp(
    dims: list[int] = [4, 16, 2],
    log_w: int = 0,
    n: int = 1,
    total_steps: int = 300,
):
    """Test that compiled backdoor MLP correctly recovers weights.

    Args:
        dims: MLP dimensions [input, hidden..., output]. Default [4, 16, 2] for 4-input XOR.
        log_w: Keccak log_w parameter. Default 0 (smallest).
        n: Keccak n parameter. Default 1.
        total_steps: Number of training steps. Default 300.
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    n_inputs = dims[0]

    # Create synthetic data (XOR/parity task - 50% random baseline)
    x_train, x_test, y_train, y_test = create_synthetic_data(500, n_inputs, device)

    # Train small ternary MLP (XOR needs more capacity and training)
    model = TernaryMLP(dims).to(device)
    warmup_steps = total_steps // 3
    model = train_ternary_mlp(model, x_train, y_train, batch_size=32, warmup_steps=warmup_steps, total_steps=total_steps)

    baseline_acc = validate_ternary(model, x_test, y_test)
    print(f"Baseline ternary accuracy: {baseline_acc:.4f}")

    # Create backdoor
    # Note: capacity c must be < b where b = 25 * 2^log_w
    # For log_w=0: b=25, so c < 25
    max_c = 25 * (2 ** log_w) - 1
    k = Keccak(log_w=log_w, n=n, c=min(10, max_c), pad_char="_")
    trigger = k.format("secret_trigger", clip=True).bitlist[:20]
    nontrigger = k.format("wrong_trigger!", clip=True).bitlist[:len(trigger)]
    dummy_trigger = k.format("dummy_trigger!", clip=True).bitlist[:len(trigger)]

    model_payloads = mlp_to_bitlists(model)
    bd_fn = get_stacked_backdoor(trigger, payloads=model_payloads, k=k)

    # Compile backdoor to MLP
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy_trigger)
    print(f"Backdoor MLP parameters: {sum(p.numel() for p in bd_mlp.parameters()):,}")

    shapes = [w.shape for w in model.extract_weights()]

    # Test with correct trigger via compiled MLP
    trigger_t = bitlist_to_tensor_w_bos(trigger, device).unsqueeze(0)
    with t.inference_mode():
        bd_mlp_out = bd_mlp(trigger_t)
    w_ternary = decode_mlp_output(bd_mlp_out, shapes, device)
    trigger_acc = validate_exec(w_ternary, x_test, y_test)
    print(f"Compiled MLP trigger accuracy: {trigger_acc:.4f}")

    # Test with wrong trigger via compiled MLP
    nontrigger_t = bitlist_to_tensor_w_bos(nontrigger, device).unsqueeze(0)
    with t.inference_mode():
        bd_mlp_out_nt = bd_mlp(nontrigger_t)
    w_ternary_nt = decode_mlp_output(bd_mlp_out_nt, shapes, device)
    nontrigger_acc = validate_exec(w_ternary_nt, x_test, y_test)
    print(f"Compiled MLP non-trigger accuracy: {nontrigger_acc:.4f}")

    # Assertions
    assert trigger_acc >= baseline_acc - 0.05, (
        f"Trigger accuracy {trigger_acc} should be close to baseline {baseline_acc}"
    )
    # Non-trigger should be close to random (50% for XOR task)
    assert nontrigger_acc < trigger_acc - 0.1, (
        f"Non-trigger accuracy {nontrigger_acc} should be much lower than trigger {trigger_acc}"
    )
    assert abs(nontrigger_acc - 0.5) < 0.2, (
        f"Non-trigger accuracy {nontrigger_acc} should be close to random (0.5)"
    )

    print("Compiled backdoor MLP test PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test backdoor encoding of ternary MLP weights")
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 16, 2],
                        help="MLP dimensions [input, hidden..., output]. Default: 4 16 2")
    parser.add_argument("--log-w", type=int, default=0,
                        help="Keccak log_w parameter. Default: 0")
    parser.add_argument("--n", type=int, default=1,
                        help="Keccak n parameter. Default: 1")
    parser.add_argument("--skip-compiled", action="store_true",
                        help="Skip the compiled MLP test (faster)")
    parser.add_argument("--total-steps", type=int, default=300,
                        help="Number of training steps. Default: 300")
    args = parser.parse_args()

    print(f"Configuration: dims={args.dims}, log_w={args.log_w}, n={args.n}, "
          f"total_steps={args.total_steps}")
    print()

    print("=" * 60)
    print("Running test_backdoor_trigger_recovery...")
    print("=" * 60)
    test_backdoor_trigger_recovery(
        dims=args.dims,
        log_w=args.log_w,
        n=args.n,
        total_steps=args.total_steps,
    )
    print()

    if not args.skip_compiled:
        print("=" * 60)
        print("Running test_backdoor_compiled_mlp...")
        print("=" * 60)
        test_backdoor_compiled_mlp(
            dims=args.dims,
            log_w=args.log_w,
            n=args.n,
            total_steps=args.total_steps,
        )
        print()

    print("All backdoor tests passed!")
