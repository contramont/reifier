"""Step 10, checkpoints: the files of a model, as Hugging Face `transformers` reads them.

A checkpoint is a directory. Its config.json describes the model. Its model.safetensors
holds the tensors by name: first 8 bytes that give the length of a header, then the header,
which is JSON with the type, shape and place of each tensor, then the numbers. Gemma's
numbers are bfloat16, bf16 for short: the upper 16 bits of a float32, which keep 8
significant bits. `fetch`, `metadata` and `read_weights` need the package huggingface_hub,
and they download from huggingface.co what is not yet on this machine.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REVISIONS = {  # the models of huggingface.co that reifier can use, each at one revision
    "unsloth/gemma-3-270m": "edcf1fe394ee1168e7c5538afd104cfb1f7947e6",
    "google/gemma-4-E2B": "433813a669639fd7eaa83155828e9000ed357d53",
}
CHUNK = 1 << 20  # `write` handles this many numbers at a time, and skips them if all are zero


@dataclass
class Metadata:
    """What a checkpoint says about its model, without the numbers."""

    shapes: dict[str, list[int]]  # the shape of each tensor, by name, in the order of the file
    config: dict  # the contents of config.json


def fetch(repo: str, name: str) -> Path:
    """The file `name` of the model `repo`, at its revision in REVISIONS."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, name, revision=REVISIONS[repo]))


def cache_file(repo: str) -> Path:
    """Where `metadata` keeps what it fetched of `repo`."""
    return Path("~/.cache/reifier", repo, REVISIONS[repo] + ".json").expanduser()


def metadata(repo: str) -> Metadata:
    """The metadata of the checkpoint of `repo`. The first call fetches the header of
    model.safetensors, not its numbers, and keeps the result in `cache_file`."""
    path = cache_file(repo)
    if not path.exists():
        from huggingface_hub import get_safetensors_metadata

        files = get_safetensors_metadata(repo, revision=REVISIONS[repo]).files_metadata
        tensors = files["model.safetensors"].tensors
        shapes = {name: tensor.shape for name, tensor in tensors.items()}
        config = json.loads(fetch(repo, "config.json").read_text())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([shapes, config]))
    return Metadata(*json.loads(path.read_text()))


def read_weights(repo: str) -> dict[str, np.ndarray]:
    """The tensors of the checkpoint of `repo`, by name, as float32. All must be bf16."""
    path = fetch(repo, "model.safetensors")
    with open(path, "rb") as file:
        header_length = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_length))
    header.pop("__metadata__", None)  # not a tensor (see `write`)
    numbers = np.memmap(path, dtype="<u2", mode="r", offset=8 + header_length)
    weights = {}
    for name, tensor in header.items():
        assert tensor["dtype"] == "BF16", f"{name} is stored as {tensor['dtype']}, not as BF16"
        start, end = tensor["data_offsets"]  # in bytes, and a bf16 takes two
        upper_bits = numbers[start // 2 : end // 2].astype(np.uint32) << 16
        weights[name] = upper_bits.view(np.float32).reshape(tensor["shape"])
    return weights


def round_to_bf16(x) -> np.ndarray:
    """x rounded to the nearest bf16, as float32: a tie goes to the one whose last bit is 0."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    last_kept_bit = bits >> 16 & 1
    # The lower 16 bits are dropped. Adding 0x7FFF first rounds up if they are more than half
    # of the last kept bit. Adding that bit too rounds up if they are exactly half and the bit
    # is 1: the last kept bit is then 0.
    return ((bits + 0x7FFF + last_kept_bit) & 0xFFFF0000).view(np.float32)


def write(out_dir, config: dict, shapes: dict, weights: dict, names: dict):
    """Write a checkpoint to the directory `out_dir`: config.json with `config`, and
    model.safetensors with every tensor of `shapes` as bf16. A tensor is that of `weights`,
    {name: array}, or zeros if `weights` has none. `names` goes to names.json: it is for
    people, and `transformers` does not read it."""
    header = {"__metadata__": {"format": "pt"}}  # as `transformers` writes it: tensors for PyTorch
    end = 0  # where the tensors so far end, in bytes after the header
    for name, shape in shapes.items():
        length = 2 * math.prod(shape)  # in bytes: a bf16 takes two
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [end, end + length]}
        end += length
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)  # the numbers start at a multiple of 8 bytes
    numbers_start = 8 + len(encoded)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config))
    (out_dir / "names.json").write_text(json.dumps(names))
    with open(out_dir / "model.safetensors", "wb") as file:
        file.write(len(encoded).to_bytes(8, "little") + encoded)
        for name, tensor in weights.items():
            flat = tensor.reshape(-1)
            tensor_start = numbers_start + header[name]["data_offsets"][0]
            for chunk_start in range(0, len(flat), CHUNK):
                chunk = flat[chunk_start : chunk_start + CHUNK]
                if chunk.any():  # zeros are skipped: a hole in a file reads as zeros
                    file.seek(tensor_start + 2 * chunk_start)
                    upper_bits = round_to_bf16(chunk).view(np.uint32) >> 16
                    file.write(upper_bits.astype("<u2").tobytes())
        file.truncate(numbers_start + end)  # the full length, even if the last numbers were skipped
