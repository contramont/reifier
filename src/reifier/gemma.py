"""google/gemma-4-E4B as a target: a program compiled into the weights of its checkpoint.

decoder.transformer compiles the program for the architecture the checkpoint's config gives,
and save writes the weights with the checkpoint's names, shapes and dtype, so that transformers
loads them as google/gemma-4-E4B: greedy generation then maps each token 0x10000 + n to the
next.
"""

import json
import math
from pathlib import Path

import numpy as np

from .core import Graph
from .decoder import Architecture, round_to_bf16, transformer

REPO, REVISION = "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"
PREFIX = "model.language_model."
CACHE_FILE = Path("~/.cache/reifier").expanduser() / f"gemma-4-E4B-{REVISION}.json"


def checkpoint_metadata(cache_file: Path = CACHE_FILE) -> tuple[dict, dict]:
    """The shapes of the tensors of the checkpoint of REPO at REVISION, and its text config: from
    huggingface.co the first time, with huggingface_hub, and then from cache_file"""
    if not cache_file.exists():
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        meta = get_safetensors_metadata(REPO, revision=REVISION).files_metadata["model.safetensors"]
        config = Path(hf_hub_download(REPO, "config.json", revision=REVISION)).read_text()
        shapes = {k: t.shape for k, t in meta.tensors.items()}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps([shapes, json.loads(config)["text_config"]]))
    return tuple(json.loads(cache_file.read_text()))


def save(program: Graph, out_dir: Path, cache_file: Path = CACHE_FILE):
    """program, compiled, in out_dir/model.safetensors, with the names, shapes and dtype (bf16)
    of the checkpoint of REPO at REVISION, which cache_file keeps (see checkpoint_metadata).
    Chunks of zeros are left as holes in the file, which read as zeros"""
    shapes, config = checkpoint_metadata(cache_file)
    w = {PREFIX + k: v for k, v in transformer(program, Architecture.from_config(config)).items()}
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "model.safetensors", "wb") as file:
        file.write(len(head).to_bytes(8, "little") + head)
        for k, v in w.items():
            if list(v.shape) != list(shapes[k]):
                raise ValueError(f"{k} is {v.shape}, but the checkpoint's is {shapes[k]}")
            at, v = 8 + len(head) + header[k]["data_offsets"][0], v.reshape(-1)
            for i in range(0, len(v), 1 << 20):
                if (chunk := v[i : i + (1 << 20)]).any():
                    file.seek(at + 2 * i)
                    file.write(
                        (round_to_bf16(chunk).view(np.uint32) >> 16).astype(np.uint16).tobytes()
                    )
        file.truncate(8 + len(head) + end)
