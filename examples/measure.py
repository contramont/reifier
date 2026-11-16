"""How well the sponge of alongside.py and Gemma get on in one checkpoint: a measure in a minute.

Run: uv run --with torch --with transformers examples/measure.py out [files of text]
on out, which examples/alongside.py wrote. It runs the model on a <bos> and 1,023 tokens of
text at a time, this repository's own files if none are given, and then on random tokens, and
says, of each:

- where the sponge speaks, how often the model's token is its token, and by how much at least;
- where it is silent, how often the model's token is that of Gemma alone, as wide and as deep
  (see beside.py: out/alone, which this writes once), and how far its logits are from those;
- how often Gemma alone, as wide, gives the token that Gemma itself gives: bf16's own noise;
- of each block beside Gemma's, in powers of two, what it reads a bit as, and what it adds
  to the stream, of what Gemma's block adds;
- the room there is, in powers of two: for a block to read a bit as less and still as 1, for
  the sponge's token to gain less and still 128 on any other, and for a block to add more and
  still under 2**-12 of what Gemma's adds, which the norm after them rounds away.
"""

import json
import sys
from pathlib import Path

import alongside
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reifier import beside, gemma

REPO = "unsloth/gemma-3-270m"
out, files = Path(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
files = files or sorted(Path(__file__).parent.parent.glob("[!.]*/**/*.py")) + [Path("README.md")]
device = "cuda" if torch.cuda.is_available() else "cpu"
if not (out / "alone").exists():
    beside.save(None, out / "alone")
tokenizer = AutoTokenizer.from_pretrained(REPO, revision=gemma.REVISIONS[REPO])
text = tokenizer("\n".join(f.read_text() for f in files), add_special_tokens=False)["input_ids"]
text = np.array(text[: (16 if device == "cuda" else 1) * 1023]).reshape(-1, 1023)  # prompts
names = json.loads((out / "names.json").read_text())["features"][-1]
program = alongside.program()
ones = [f for f, s in enumerate(names) if s == "ONE"][1:]  # block d reads ones[d]
load = AutoModelForCausalLM.from_pretrained
both, alone = load(out).to(device), load(out / "alone").to(device)
itself = load(REPO, revision=gemma.REVISIONS[REPO]).to(device)
half, seen = both.config.hidden_size // 2, {}  # what the hooks keep of the last run


def reads(d):  # the ONE of block d, as the block reads it
    return lambda _, __, x: seen.__setitem__(("reads", d), x[..., ones[d]].float().abs())


def adds(d):  # what block d gives, over what Gemma's block gives
    ours = lambda x: x[..., :half].float().norm(dim=-1) / x[..., half:].float().norm(dim=-1)
    return lambda _, __, x: seen.__setitem__(("adds", d), ours(x))


blocks = range(beside.OWN, len(ones) - 1)  # those beside Gemma's
for d in blocks:
    both.model.layers[d].pre_feedforward_layernorm.register_forward_hook(reads(d))
    both.model.layers[d].mlp.register_forward_hook(adds(d))


def measure(ids: np.ndarray, title: str):
    ids = np.c_[np.full(len(ids), tokenizer.bos_token_id), ids]
    bits = np.array([alongside.CODES[t] for t in ids.ravel()]).reshape(*ids.shape, -1)
    older = np.concatenate([bits[:, :1], bits[:, :-1]], 1)  # position 0 reads its token twice
    values = program.values(np.concatenate([older, bits], -1).reshape(-1, 2 * bits.shape[-1]))
    code = np.stack([values[y] for y in program.outputs], -1).reshape(bits.shape)
    speaks = torch.tensor((code != 0.5).all(-1), device=device)
    token = code.round().astype(int) @ (1 << np.arange(bits.shape[-1]))  # of that code
    token = torch.tensor(token, device=device)
    tops, room, gains, off = [], {}, [], []
    for i in range(0, len(ids), 4):  # 4 prompts at a time: the logits are large
        batch, say, its = (x[i : i + 4] for x in (torch.tensor(ids, device=device), speaks, token))
        with torch.no_grad():
            logits = [model(batch).logits.float() for model in (both, alone, itself)]
        tops.append(torch.stack([x.argmax(-1) for x in logits]))
        best = logits[0].topk(2).values  # the sponge's token, over the best of the others
        other = torch.where(tops[-1][0] == its, best[..., 1], best[..., 0])
        gains.append((logits[0].gather(-1, its[..., None])[..., 0] - other)[say])
        off.append((logits[0] - logits[1]).abs().amax(-1)[~say])
        for k, v in seen.items():
            room.setdefault(k, []).append(v[:, 1:].flatten().log2())  # after the <bos>
    (mine, wide, real), gain, off = torch.cat(tops, 1), torch.cat(gains).min(), torch.cat(off)
    right, same = (mine == token)[speaks], (mine == wide)[~speaks]
    print(f"\n{title}: {len(ids)} prompts of {ids.shape[1]} tokens, {device}, {both.dtype}")
    print(f"  the sponge speaks at {right.numel()} positions: the model's token is its token at")
    print(f"    {right.sum()} ({right.float().mean():.2%}), by {gain:.3g} at least in the logits")
    print(f"  it is silent at {same.numel()}: the model's token is Gemma's, alone as wide, at")
    print(f"    {same.sum()} ({same.float().mean():.2%}); the logit furthest from Gemma's is")
    print(f"    {off.median():.2g} from it as a rule, {off.max():.2g} at most")
    print(f"  Gemma alone, as wide, gives Gemma's own token at {(wide == real).float().mean():.2%}")
    print("  block: reads a bit as 2** (least, usual, most) | adds 2** of Gemma's block (most)")
    least, most = [], []
    for d in blocks:
        read, add = torch.cat(room["reads", d]), torch.cat(room["adds", d]).max()
        least.append(read.min()), most.append(add)
        print(f"  {d:5d}: {read.min():10.1f} {read.median():6.1f} {read.max():6.1f} | {add:6.1f}")
    print(f"  room, in powers of two: {min(least):.1f} to read a bit as less, "
          f"{gain.log2() - 7:.1f} for its token to gain less, {-12 - max(most):.1f} to add more")  # fmt: skip


measure(text, f"text of {len(files)} files, from {files[0]}")
measure(np.random.default_rng(0).integers(0, 2**18, text.shape), "random tokens")
