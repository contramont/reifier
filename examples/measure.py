"""How well the sponge of alongside.py and Gemma get on in one checkpoint, how large the sponge
is beside Gemma, and how much noise each takes: a measure in a minute, on a GPU.

Run: uv run --with torch --with transformers examples/measure.py out [files of text]
on out, which examples/alongside.py wrote. It runs the model on a <bos> and 1,023 tokens of
text at a time, up to 16 times on a GPU and once without: the start of the files given, or of
those of src/reifier, and then as many random tokens. The sponge speaks where the program of
alongside.py gives a code, of 144 bits: its token is the one whose number is the first 18.
Elsewhere each of them is 1/2, and it is silent. Of each run, this prints:

- where the sponge speaks, how often the model's token is its token, by how much at least in
  the logits, and what that token's logit is as a rule, the median, and at most;
- where it is silent, how often the model's token is that of Gemma alone, as wide (see
  beside.py: out/alone, which this writes once), and how far its logits are from those;
- how often Gemma alone gives Gemma's own token: bf16's own noise, the floor for the line above;
- the largest entry after the <bos>, of either sign, and its layer: of the stream, of what a
  block reads after its norm, of its gates, values and hidden units and of what it gives, in
  the sponge's own layers, where the stream is its features, and in Gemma's; and of the
  model's logits, which are the sponge's where it speaks, and of those of Gemma alone.

Then, of each kind of tensor, the largest weight of the sponge's and of Gemma's own, a norm's
weight being what it multiplies by. And under each noise of `noisy` on the weights, in three
draws, how often on the text the sponge's token is still the model's, how often the model's is
still that of Gemma alone with no noise, where the sponge is silent, and how often Gemma
alone's is, at every position.
"""

import copy
import json
import sys
from functools import partial
from pathlib import Path

import alongside
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reifier import beside, gemma

REPO = "unsloth/gemma-3-270m"
NOISE = {"times": (0.02, 0.05, 0.07, 0.1), "plus": (0.02, 0.05, 0.1), "bits": (8, 6, 5)}  # levels
NOISE |= {"zeros": (0.001, 0.01), "norms": (0.0003, 0.001)}  # of each noise of `noisy`
WHAT = "stream", "reads", "gate", "value", "hidden", "gives", "logit"  # what the hooks see, on
MODULES = "", "pre_feedforward_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp"
out, files = Path(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
files = files or sorted(Path(__file__).parent.parent.glob("src/reifier/*.py"))
device = "cuda" if torch.cuda.is_available() else "cpu"
if not (out / "alone").exists():
    beside.save(None, out / "alone")
tokenizer = AutoTokenizer.from_pretrained(REPO, revision=gemma.REVISIONS[REPO])
text = tokenizer("\n".join(f.read_text() for f in files), add_special_tokens=False)["input_ids"]
text = np.reshape(text[: len(text) // 1023 * 1023], (-1, 1023))[: 16 if device == "cuda" else 1]
own = [bool(units) for units in json.loads((out / "names.json").read_text())["hidden"]]
program, seen = alongside.program(), {}  # own[i]: whether layer i is the sponge's own, and not
torch.set_grad_enabled(False)  # Gemma's; seen: what the hooks keep of the last run
both, alone = (AutoModelForCausalLM.from_pretrained(d).to(device) for d in (out, out / "alone"))
itself = AutoModelForCausalLM.from_pretrained(REPO, revision=gemma.REVISIONS[REPO]).to(device)


def hook(what: str, mine: bool, i: int, module, args, x):
    """A forward hook on a module of layer i, the sponge's own or not: seen[what, mine] is the
    largest entry after the <bos> of what the module gives, or reads if that is a hidden unit,
    and i. Of the stream, it is of the features of whose layer it is: the sponge's are the first"""
    x = args[0] if what == "hidden" else x[0] if isinstance(x, tuple) else x
    x = x.chunk(2, -1)[not mine] if what == "stream" else x
    large = x[:, 1:].abs().max().item(), i
    seen[what, mine] = max(large, seen.get((what, mine), large))


def measure(ids: np.ndarray, title: str):
    """Prints how the sponge and Gemma get on on the prompts ids, a <bos> before each, and what
    the hooks saw. Gives the prompts, where the sponge speaks, its token, and Gemma's, alone"""
    ids = np.c_[np.full(len(ids), tokenizer.bos_token_id), ids]
    bits = np.array([alongside.CODES[t] for t in ids.ravel()]).reshape(*ids.shape, -1)
    older = np.concatenate([bits[:, :1], bits[:, :-1]], 1)  # position 0 reads its token twice
    values = program.values(np.concatenate([older, bits], -1).reshape(-1, 2 * bits.shape[-1]))
    code = np.stack([values[y] for y in program.outputs], -1).reshape(bits.shape)
    number = code[..., : alongside.BITS].round().astype(int) @ (1 << np.arange(alongside.BITS))
    said = (code != 0.5).all(-1), number  # where it speaks, and the token of that code
    ids, speaks, token = (torch.tensor(x, device=device) for x in (ids, *said))
    tops, gains, off = [], [], []
    seen.clear()
    for i in range(0, len(ids), 2):  # 2 prompts at a time: the logits are large
        logits = [model(ids[i : i + 2]).logits.float() for model in (both, alone, itself)]
        tops.append(torch.stack([x.argmax(-1) for x in logits]))
        its = token[i : i + 2, :, None]  # the sponge's token: its logit, and that over the
        logit = logits[0].gather(-1, its)  # best other
        gain = logit - logits[0].scatter(-1, its, -torch.inf).amax(-1, True)
        gains.append(torch.cat([logit, gain], -1)[speaks[i : i + 2]])
        off.append((logits[0] - logits[1]).abs_().amax(-1))
    (mine, wide, real), (logit, gain), off = torch.cat(tops, 1), torch.cat(gains).T, torch.cat(off)
    right, same, off = (mine == token)[speaks], (mine == wide)[~speaks], off[~speaks]
    usual, most, least = logit.median(), logit.max(), gain.min()
    print(f"\n{title}: {len(ids)} prompts of {ids.shape[1]} tokens, {device}, {both.dtype}")
    print(f"  the sponge speaks at {right.numel()} positions: the model's token is its token at")
    print(f"    {right.sum()} ({right.float().mean():.2%}), by {least:.3g} at least in the logits;")
    print(f"    its logit is {usual:.3g} as a rule, {most:.3g} at most")
    print(f"  it is silent at {same.numel()}: the model's token is Gemma's, alone as wide, at")
    print(f"    {same.sum()} ({same.float().mean():.2%}); the logit furthest from Gemma's is")
    print(f"    {off.median():.2g} from it as a rule, {off.max():.2g} at most")
    print(f"  Gemma alone, as wide, gives Gemma's own token at {(wide == real).float().mean():.2%}")
    print("  the largest after the <bos> (its layer): in the sponge's layers | in Gemma's")
    for what in WHAT:
        cells = ["{:9.3g} ({:2d})".format(*seen[what, whose]) for whose in (True, False)]
        print(f"    {what:38s}", " | ".join(cells))
    return ids, speaks, token, wide


def noisy(model, how: str, s: float, draw: int):
    """What a copy of model gives after each position of the text with noise s e on its weights,
    e normal, by its draw. times: every weight w is w (1 + s e), a norm's being what it
    multiplies by. plus: every weight of a projection or the embedding, zeros too, is s e more,
    in root mean squares of Gemma's own of its kind; norms: every weight of a norm is, and no
    other. bits: every projection, and the embedding, is rounded to whole steps, 2**(s - 1) - 1
    of which are the largest weight of its tensor. zeros: each weight of those is 0 with
    probability s"""
    model, e = copy.deepcopy(model), torch.Generator(device).manual_seed(draw)
    for k, p in model.named_parameters():
        norm = k.endswith("norm.weight")
        w = p.float() + norm
        if how == "times":
            w *= 1 + s * torch.randn(w.shape, generator=e, device=device)
        elif how == ("norms" if norm else "plus"):
            w += s * rms[k.split(".")[-2]] * torch.randn(w.shape, generator=e, device=device)
        elif how == "bits" and not norm and w.any():  # a tensor of zeros stays zeros
            step = w.abs().max() / (2 ** (s - 1) - 1)
            w = (w / step).round() * step
        elif how == "zeros" and not norm:
            w *= torch.rand(w.shape, generator=e, device=device) >= s
        p.copy_(w - int(norm))
    return torch.cat([model(ids[i : i + 4]).logits.argmax(-1) for i in range(0, len(ids), 4)])


for i, layer in enumerate(both.model.layers):
    for what, module in zip(WHAT, MODULES):
        layer.get_submodule(module).register_forward_hook(partial(hook, what, own[i], i))
for model in both, alone:  # the logits come after the last layer: the model's, and Gemma's alone
    model.lm_head.register_forward_hook(partial(hook, "logit", model is both, len(own)))
title = f"the first {text.size} tokens of {len(files)} files, {files[0]} first"
ids, speaks, token, clean = measure(text, title)
measure(np.random.default_rng(0).integers(0, 2**18, text.shape), "random tokens")
trained, largest = {}, {}  # by kind of tensor: Gemma's own weights, and the sponge's largest
for k, v in itself.named_parameters():
    trained.setdefault(k.split(".")[-2], []).append(v.float() + k.endswith("norm.weight"))
for k, v in both.named_parameters():  # the sponge's are all of a layer of its own, and of the
    i, kind = k.split(".")[2], k.split(".")[-2]  # others all but Gemma's: the last of each axis
    v = (v.float() + k.endswith("norm.weight")).abs()
    if not (i.isdigit() and own[int(i)]):
        v[tuple(slice(-d, None) for d in trained[kind][0].shape)] = 0
    largest[kind] = max(largest.get(kind, 0.0), v.max().item())
trained = {k: torch.cat([x.flatten() for x in v]) for k, v in trained.items()}
rms = {k: v.square().mean().sqrt() for k, v in trained.items()}
print("\nthe largest weight of each kind: the sponge's | Gemma's own | the first over the second")
for k, v in trained.items():
    print(f"  {k:30s} {largest[k]:9.3g} | {v.abs().max():9.3g} | {largest[k] / v.abs().max():.2f}")
print("\nnoise on the weights, on the text, in 3 draws: the sponge's token is the model's | the")
print("model's is that of Gemma alone with no noise, where the sponge is silent | Gemma alone's is")
for how, levels in NOISE.items():
    draws = range(1 if how == "bits" else 3)  # rounding is the same in each
    for s in levels:
        mine, wide = (torch.stack([noisy(m, how, s, d) for d in draws]) for m in (both, alone))
        still = (mine == token)[:, speaks], (mine == clean)[:, ~speaks], wide == clean
        print(f"  {how:5s} {s:<6g}", " | ".join(f"{x.float().mean():.4f}" for x in still))
