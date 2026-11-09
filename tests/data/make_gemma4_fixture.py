"""Write gemma4_tiny.npz: the logits of transformers' Gemma 4 for random weights of a tiny
config, which tests/test_decoder.py checks reifier.decoder.forward against.

The config has sliding and full-attention layers, a window shorter than the tokens, layers that
reuse keys and values, and grouped heads; the per-layer inputs are zero, as forward assumes.
Run: uv run --no-project --with torch==2.15.0 --with transformers==5.22.2 tests/data/make_gemma4_fixture.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

CONFIG = {
    "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 6,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "global_head_dim": 16,
    "layer_types": ["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2
    + ["full_attention"],
    "num_kv_shared_layers": 2, "sliding_window": 5, "vocab_size": 32,
    "vocab_size_per_layer_input": 32, "hidden_size_per_layer_input": 4,
    "final_logit_softcapping": 30.0, "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh",
    "attention_k_eq_v": False, "enable_moe_block": False, "tie_word_embeddings": True,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                           "rope_theta": 1e6},
    },
    "bos_token_id": 2, "pad_token_id": 0, "eos_token_id": 1,
}  # fmt: skip

if __name__ == "__main__":
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**CONFIG)).float().eval()
    with torch.no_grad():
        for name, t in model.state_dict().items():
            if name.endswith("layer_scalar"):
                t.fill_(0.9)
            elif t.is_floating_point():
                t.copy_(torch.randn_like(t) * (0.3 if t.dim() > 1 else 1.0))
            if "post_per_layer_input_norm" in name:
                t.zero_()  # the per-layer inputs then add nothing
    ids = np.random.default_rng(0).integers(0, CONFIG["vocab_size"], 13)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(ids)[None]).logits[0].numpy()
    weights = {
        k.removeprefix("model."): v.numpy()
        for k, v in model.state_dict().items()
        if k.startswith("model.") and "per_layer" not in k
    }
    np.savez_compressed(
        Path(__file__).parent / "gemma4_tiny.npz",
        config=json.dumps(CONFIG),
        ids=ids,
        logits=logits,
        **{f"w:{k}": v for k, v in weights.items()},
    )
