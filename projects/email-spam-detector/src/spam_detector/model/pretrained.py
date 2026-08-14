"""
Utilities for loading pretrained OpenAI GPT-2 weights.

The GPT architecture itself is implemented in gpt.py.

This module performs three tasks:

1. Download the original GPT-2 checkpoint files.
2. Convert the TensorFlow checkpoint parameters into the structure
   expected by our PyTorch GPTModel.
3. Copy the pretrained weights into our model.

The implementation follows chapter 5 of
"Build a Large Language Model (From Scratch)".
"""

from pathlib import Path

import numpy as np
import torch

from spam_detector.model.gpt import (
    GPT2_SMALL_CONFIG,
    GPTModel,
)
from spam_detector.model.gpt_download import (
    download_and_load_gpt2,
)
from spam_detector.paths import PROJECT_ROOT

# Pretrained-model configuration


GPT2_MODEL_SIZE = "124M"

GPT2_MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "gpt2"
)



# Weight assignment helper


def assign(
    left: torch.Tensor,
    right: np.ndarray,
) -> torch.nn.Parameter:
    """
    Convert an OpenAI GPT-2 weight array into a PyTorch parameter.

    The function first verifies that the source and destination
    shapes are identical.

    This catches architecture mismatches before invalid weights
    can be inserted into the model.
    """

    if left.shape != right.shape:
        raise ValueError(
            "Shape mismatch. "
            f"Model shape: {left.shape}, "
            f"pretrained shape: {right.shape}"
        )

    tensor = torch.tensor(
        right,
        dtype=left.dtype,
        device=left.device,
    )

    return torch.nn.Parameter(
        tensor
    )



# Transfer OpenAI weights into GPTModel


def load_weights_into_gpt(
    gpt: GPTModel,
    params: dict,
) -> None:
    """
    Copy pretrained OpenAI GPT-2 weights into our GPTModel.

    OpenAI's TensorFlow implementation uses different parameter
    names and tensor layouts than our PyTorch implementation.

    This function maps those parameters into the corresponding
    layers of GPTModel.
    """

    
    # Embeddings
    

    gpt.pos_emb.weight = assign(
        gpt.pos_emb.weight,
        params["wpe"],
    )

    gpt.tok_emb.weight = assign(
        gpt.tok_emb.weight,
        params["wte"],
    )

    
    # Transformer blocks
    

    if len(params["blocks"]) != len(
        gpt.trf_blocks
    ):
        raise ValueError(
            "Number of pretrained Transformer blocks "
            "does not match the GPT model."
        )

    for block_index in range(
        len(params["blocks"])
    ):
        block_params = (
            params["blocks"][
                block_index
            ]
        )

        block = gpt.trf_blocks[
            block_index
        ]

        
        # Query, key, and value projection weights
        
        
        # OpenAI stores Q, K, and V together in c_attn.
        # We split the combined matrix into three parts.

        q_w, k_w, v_w = np.split(
            block_params[
                "attn"
            ][
                "c_attn"
            ][
                "w"
            ],
            3,
            axis=-1,
        )

        block.att.W_query.weight = assign(
            block.att.W_query.weight,
            q_w.T,
        )

        block.att.W_key.weight = assign(
            block.att.W_key.weight,
            k_w.T,
        )

        block.att.W_value.weight = assign(
            block.att.W_value.weight,
            v_w.T,
        )

        
        # Query, key, and value biases
        

        q_b, k_b, v_b = np.split(
            block_params[
                "attn"
            ][
                "c_attn"
            ][
                "b"
            ],
            3,
            axis=-1,
        )

        if (
            block.att.W_query.bias
            is None
        ):
            raise ValueError(
                "GPT model has no QKV bias. "
                "Use qkv_bias=True when loading "
                "OpenAI GPT-2 weights."
            )

        block.att.W_query.bias = assign(
            block.att.W_query.bias,
            q_b,
        )

        block.att.W_key.bias = assign(
            block.att.W_key.bias,
            k_b,
        )

        block.att.W_value.bias = assign(
            block.att.W_value.bias,
            v_b,
        )

        
        # Attention output projection
        

        block.att.out_proj.weight = assign(
            block.att.out_proj.weight,
            block_params[
                "attn"
            ][
                "c_proj"
            ][
                "w"
            ].T,
        )

        block.att.out_proj.bias = assign(
            block.att.out_proj.bias,
            block_params[
                "attn"
            ][
                "c_proj"
            ][
                "b"
            ],
        )

        
        # Feed-forward network
        

        block.ff.layers[
            0
        ].weight = assign(
            block.ff.layers[
                0
            ].weight,
            block_params[
                "mlp"
            ][
                "c_fc"
            ][
                "w"
            ].T,
        )

        block.ff.layers[
            0
        ].bias = assign(
            block.ff.layers[
                0
            ].bias,
            block_params[
                "mlp"
            ][
                "c_fc"
            ][
                "b"
            ],
        )

        block.ff.layers[
            2
        ].weight = assign(
            block.ff.layers[
                2
            ].weight,
            block_params[
                "mlp"
            ][
                "c_proj"
            ][
                "w"
            ].T,
        )

        block.ff.layers[
            2
        ].bias = assign(
            block.ff.layers[
                2
            ].bias,
            block_params[
                "mlp"
            ][
                "c_proj"
            ][
                "b"
            ],
        )

        
        # Layer normalization
        

        block.norm1.scale = assign(
            block.norm1.scale,
            block_params[
                "ln_1"
            ][
                "g"
            ],
        )

        block.norm1.shift = assign(
            block.norm1.shift,
            block_params[
                "ln_1"
            ][
                "b"
            ],
        )

        block.norm2.scale = assign(
            block.norm2.scale,
            block_params[
                "ln_2"
            ][
                "g"
            ],
        )

        block.norm2.shift = assign(
            block.norm2.shift,
            block_params[
                "ln_2"
            ][
                "b"
            ],
        )

    
    # Final LayerNorm
    

    gpt.final_norm.scale = assign(
        gpt.final_norm.scale,
        params["g"],
    )

    gpt.final_norm.shift = assign(
        gpt.final_norm.shift,
        params["b"],
    )

    
    # Language-model output head
    
    #
    # GPT-2 uses the token embedding weights for the output
    # projection as well (weight tying).
    #
    # For our purposes we copy the pretrained values here.
    # classifier.py will replace this layer later anyway.

    gpt.out_head.weight = assign(
        gpt.out_head.weight,
        params["wte"],
    )



# Create pretrained GPT-2 small


def create_pretrained_gpt2_small(
    models_dir: Path = GPT2_MODELS_DIR,
) -> GPTModel:
    """
    Download and create the pretrained GPT-2 small model.

    On the first call, the original GPT-2 checkpoint files are
    downloaded.

    On later calls, already downloaded files are reused.
    """

    models_dir = Path(
        models_dir
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading pretrained GPT-2 "
        f"{GPT2_MODEL_SIZE} weights..."
    )

    # Download and read the original GPT-2 checkpoint.
    settings, params = (
        download_and_load_gpt2(
            model_size=GPT2_MODEL_SIZE,
            models_dir=str(
                models_dir
            ),
        )
    )

    
    # Verify that the checkpoint matches our configuration
    

    expected_settings = {
        "n_vocab":
            GPT2_SMALL_CONFIG[
                "vocab_size"
            ],
        "n_ctx":
            GPT2_SMALL_CONFIG[
                "context_length"
            ],
        "n_embd":
            GPT2_SMALL_CONFIG[
                "emb_dim"
            ],
        "n_head":
            GPT2_SMALL_CONFIG[
                "n_heads"
            ],
        "n_layer":
            GPT2_SMALL_CONFIG[
                "n_layers"
            ],
    }

    for key, expected_value in (
        expected_settings.items()
    ):
        actual_value = settings[
            key
        ]

        if actual_value != expected_value:
            raise ValueError(
                f"GPT-2 setting mismatch for "
                f"{key}: expected "
                f"{expected_value}, got "
                f"{actual_value}"
            )

    
    # Create our GPT architecture


    model = GPTModel(
        GPT2_SMALL_CONFIG
    )

    
    # Replace random weights with pretrained weights
    

    load_weights_into_gpt(
        model,
        params,
    )

    model.eval()

    return model