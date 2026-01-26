"""Test the switcher model combining backdoor and benign MLPs."""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reifier.tensors.compilation import Compiler
from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.examples.backdoors import get_stacked_backdoor_with_flags
from reifier.examples.keccak import Keccak
from reifier.examples.switcher import create_switcher, normalize_flags
from reifier.neurons.core import Bit, const
from reifier.train.train_utils import map_to_relaxed_bools


# --- Ternary MLP for payload encoding ---

class Ternarize(t.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return t.sign(t.round(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(t.randn(out_features, in_features) * 0.5)

    def forward(self, x):
        return F.linear(x, Ternarize.apply(self.weight))

    def get_ternary_weights(self) -> t.Tensor:
        return t.sign(t.round(self.weight.data))


class TernaryMLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(TernaryLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def extract_weights(self) -> list[t.Tensor]:
        return [layer.get_ternary_weights() for layer in self.layers if isinstance(layer, TernaryLinear)]


# --- Helpers ---

def mlp_to_bitlists(model: TernaryMLP) -> list[list[Bit]]:
    """Convert ternary MLP weights to bitlists for backdoor encoding."""
    bitlists = []
    for w in model.extract_weights():
        bits = []
        for v in w.flatten().tolist():
            if v == -1:
                bits.extend([0, 1])
            elif v == 0:
                bits.extend([0, 0])
            else:
                bits.extend([1, 0])
        bitlists.append(const(bits))
    return bitlists


def binary_to_ternary_flat(binary: t.Tensor, shapes: list[t.Size]) -> t.Tensor:
    total_params = sum(s.numel() for s in shapes)
    ternary = t.zeros(total_params)
    for i in range(total_params):
        b0, b1 = binary[2*i], binary[2*i + 1]
        if b0 == 1 and b1 == 0:
            ternary[i] = 1
        elif b0 == 0 and b1 == 1:
            ternary[i] = -1
    return ternary


def flat_to_matrices(flat: t.Tensor, shapes: list[t.Size]) -> list[t.Tensor]:
    matrices, offset = [], 0
    for shape in shapes:
        size = shape.numel()
        matrices.append(flat[offset:offset + size].reshape(shape))
        offset += size
    return matrices


def bitlist_to_tensor_w_bos(bits: list[Bit], device) -> t.Tensor:
    return t.tensor([1.0] + [float(b.activation) for b in bits], device=device)


def create_majority_data(n_samples: int, n_inputs: int, device):
    t.manual_seed(43)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    y = (x.sum(dim=1) > n_inputs / 2).long().to(device)
    split = int(0.8 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def train_ternary_mlp(model, x_train, y_train, steps=200):
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    step = 0
    while step < steps:
        for x, y in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
            step += 1
            if step >= steps:
                break
    return model


# --- Tests ---

def test_backdoor_with_flags():
    """Test backdoor flag computation (direct evaluation)."""
    print("Testing backdoor with flags (direct evaluation)...")

    k = Keccak(log_w=0, n=1, c=10, pad_char="_")

    trigger = k.format("secret_trigger", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("wrong_trigger!", clip=True).bitlist[:len(trigger)]
    payload = const([1, 0, 1, 1, 0, 0, 1, 0])

    bd_fn = get_stacked_backdoor_with_flags(trigger, [payload], k, n_flag_bits=8)

    # Output is flat: [payload_bits..., flag_t, flag_nt]
    # Test with trigger
    result_t = bd_fn(trigger)
    flag_t, flag_nt = result_t[-2].activation, result_t[-1].activation
    assert flag_t == True, "flag_triggered should be True"
    assert flag_nt == False, "flag_not_triggered should be False"
    print(f"  Trigger: [{flag_t}, {flag_nt}] OK")

    # Test with non-trigger
    result_nt = bd_fn(nontrigger)
    flag_t_nt, flag_nt_nt = result_nt[-2].activation, result_nt[-1].activation
    assert flag_t_nt == False, "flag_triggered should be False"
    assert flag_nt_nt == True, "flag_not_triggered should be True"
    print(f"  Non-trigger: [{flag_t_nt}, {flag_nt_nt}] OK")

    print("  PASSED\n")


def test_switcher():
    """Test SwiGLU-based switcher with big/small values."""
    print("Testing switcher...")

    n_features = 4
    switcher = create_switcher(n_features)

    BOS = 1.5
    bad = t.tensor([BOS, 0.0, BOS, 0.0])
    benign = t.tensor([BOS, BOS, BOS, BOS])

    # Trigger: flag_t big, flag_nt small -> select bad
    x1 = t.cat([bad, benign, t.tensor([BOS, 0.0])]).unsqueeze(0)
    out1 = switcher(x1).squeeze()
    assert t.allclose(out1, bad, atol=0.1), f"Expected ~{bad.tolist()}, got {out1.tolist()}"
    print(f"  Trigger selects bad: OK")

    # Non-trigger: flag_t small, flag_nt big -> select benign
    x2 = t.cat([bad, benign, t.tensor([0.0, BOS])]).unsqueeze(0)
    out2 = switcher(x2).squeeze()
    assert t.allclose(out2, benign, atol=0.1), f"Expected ~{benign.tolist()}, got {out2.tolist()}"
    print(f"  Non-trigger selects benign: OK")

    print("  PASSED\n")


def test_compiled_backdoor():
    """Test compiled backdoor produces correct flags."""
    print("Testing compiled backdoor...")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train a small ternary MLP
    x_train, x_test, y_train, y_test = create_majority_data(500, 4, device)
    ternary_mlp = TernaryMLP([4, 8, 2]).to(device)
    ternary_mlp = train_ternary_mlp(ternary_mlp, x_train, y_train)

    # Create backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(ternary_mlp), k)

    # Compile
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Compiled MLP: {len(bd_mlp.layers)} layers, {sum(p.numel() for p in bd_mlp.parameters()):,} params")

    # Test trigger
    trigger_t = bitlist_to_tensor_w_bos(trigger, device)
    with t.inference_mode():
        bd_out_t = bd_mlp(trigger_t.unsqueeze(0))
    flag_t, flag_nt = normalize_flags(bd_out_t[0])
    print(f"  Trigger flags (normalized): [{flag_t:.2f}, {flag_nt:.2f}]")
    assert flag_t > flag_nt, f"Expected flag_t > flag_nt, got {flag_t} vs {flag_nt}"

    # Test non-trigger
    nontrigger_t = bitlist_to_tensor_w_bos(nontrigger, device)
    with t.inference_mode():
        bd_out_nt = bd_mlp(nontrigger_t.unsqueeze(0))
    flag_t_nt, flag_nt_nt = normalize_flags(bd_out_nt[0])
    print(f"  Non-trigger flags (normalized): [{flag_t_nt:.2f}, {flag_nt_nt:.2f}]")
    assert flag_nt_nt > flag_t_nt, f"Expected flag_nt > flag_t, got {flag_nt_nt} vs {flag_t_nt}"

    print("  PASSED\n")


def test_combined_model():
    """Test full combined model with switcher (for debugging - passes values between modules)."""
    print("Testing combined model...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train ternary MLP for backdoor payload
    x_train, x_test, y_train, y_test = create_majority_data(500, 4, device)
    ternary_mlp = TernaryMLP([4, 8, 2]).to(device)
    ternary_mlp = train_ternary_mlp(ternary_mlp, x_train, y_train)
    shapes = [w.shape for w in ternary_mlp.extract_weights()]
    n_payload_bits = sum(2 * s.numel() for s in shapes)

    # Create and compile backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(ternary_mlp), k)
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Backdoor MLP: {len(bd_mlp.layers)} layers")

    # Create switcher
    switcher = create_switcher(n_features=2)

    def combined_forward(trigger_input, task_input):
        """Run backdoor and switch based on flags."""
        # Run backdoor
        trigger_t = bitlist_to_tensor_w_bos(trigger_input, device).unsqueeze(0)
        with t.inference_mode():
            bd_out = bd_mlp(trigger_t)

        # Get normalized flags
        flag_t, flag_nt = normalize_flags(bd_out[0])

        # Decode payload and execute on task input
        relaxed = map_to_relaxed_bools(bd_out)
        w_binary = (relaxed[0, 1:1+n_payload_bits] > 0.5).float()
        w_ternary = flat_to_matrices(binary_to_ternary_flat(w_binary, shapes), shapes)

        bad_x = task_input[:4].unsqueeze(0)  # First 4 bits as input to decoded MLP
        for i, w in enumerate(w_ternary):
            bad_x = F.linear(bad_x, w.to(device))
            if i < len(w_ternary) - 1:
                bad_x = F.relu(bad_x)
        bad_output = bad_x.squeeze(0)  # Shape: [2]

        # For benign output, use a simple heuristic (majority)
        benign_output = t.tensor([0.0, 1.0] if task_input.sum() > 4 else [1.0, 0.0], device=device)

        # Switch: [bad(2), benign(2), flag_t, flag_nt] = 6 features
        switch_input = t.cat([bad_output, benign_output, flag_t.unsqueeze(0), flag_nt.unsqueeze(0)]).unsqueeze(0)
        return switcher(switch_input).squeeze(), (flag_t.item(), flag_nt.item())

    # Test
    test_x = t.tensor([1., 1., 1., 1., 0., 0., 0., 0.], device=device)

    out_t, flags_t = combined_forward(trigger, test_x)
    print(f"  Trigger: flags=[{flags_t[0]:.4f}, {flags_t[1]:.4f}]")
    assert flags_t[0] > flags_t[1], "Trigger should have flag_t > flag_nt"

    out_nt, flags_nt = combined_forward(nontrigger, test_x)
    print(f"  Non-trigger: flags=[{flags_nt[0]:.4f}, {flags_nt[1]:.4f}]")
    assert flags_nt[1] > flags_nt[0], "Non-trigger should have flag_nt > flag_t"

    print("=" * 60)
    print("  PASSED\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all", choices=["flags", "switcher", "compiled", "combined", "all"])
    args = parser.parse_args()

    if args.test in ("all", "flags"):
        test_backdoor_with_flags()

    if args.test in ("all", "switcher"):
        test_switcher()

    if args.test in ("all", "compiled"):
        test_compiled_backdoor()

    if args.test in ("all", "combined"):
        test_combined_model()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
