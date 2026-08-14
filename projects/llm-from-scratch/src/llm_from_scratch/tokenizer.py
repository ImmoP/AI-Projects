"""
Tokenizer helpers around tiktoken's GPT-2 BPE encoding, pulled out of the
notebook so both the notebook and the test suite import the same
definitions instead of two copies drifting apart. Unchanged from the
notebook version.
"""

import tiktoken
import torch


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
