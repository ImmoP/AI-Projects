"""
GPT-2 model architecture used for spam email classification.

This implementation follows the GPT architecture developed in
"Build a Large Language Model (From Scratch)".

The model is initially a standard GPT language model:

token IDs
    ↓
token embeddings + positional embeddings
    ↓
Transformer blocks
    ↓
final LayerNorm
    ↓
language-model output head

Later, classifier.py will replace the language-model output head
with a two-class classification head:

0 = ham
1 = spam
"""

import torch
import torch.nn as nn

# GPT-2 small configuration


GPT2_SMALL_CONFIG = {
    "vocab_size": 50_257,
    "context_length": 1_024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,

    # OpenAI GPT-2 uses bias terms in the query/key/value layers.
    # This must be True when loading the pretrained weights.
    "qkv_bias": True,
}



# Layer Normalization


class LayerNorm(nn.Module):
    """
    Layer normalization used by GPT-2.

    Normalization is performed over the final embedding dimension.

    The scale and shift parameters are trainable.
    """

    def __init__(
        self,
        emb_dim: int,
    ):
        super().__init__()

        self.eps = 1e-5

        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )

        self.shift = nn.Parameter(
            torch.zeros(emb_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        normalized = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )

        return (
            self.scale * normalized
            + self.shift
        )



# GELU activation


class GELU(nn.Module):
    """
    Approximate GELU activation function used by GPT-2.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(
                        torch.tensor(
                            2.0 / torch.pi,
                            device=x.device,
                        )
                    )
                    * (
                        x
                        + 0.044715
                        * torch.pow(x, 3)
                    )
                )
            )
        )



# Feed-forward network


class FeedForward(nn.Module):
    """
    Feed-forward network inside each Transformer block.

    The hidden dimension is expanded by a factor of four:

        768 -> 3072 -> 768
    """

    def __init__(
        self,
        cfg: dict,
    ):
        super().__init__()

        emb_dim = cfg["emb_dim"]

        self.layers = nn.Sequential(
            nn.Linear(
                emb_dim,
                4 * emb_dim,
            ),
            GELU(),
            nn.Linear(
                4 * emb_dim,
                emb_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.layers(x)



# Multi-head causal self-attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention used by GPT-2.

    The causal mask ensures that each token can only attend to
    itself and earlier tokens.

    Future tokens are hidden from the current token.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError(
                "d_out must be divisible by num_heads."
            )

        self.d_out = d_out
        self.num_heads = num_heads

        self.head_dim = (
            d_out // num_heads
        )

        # Linear projections for query, key, and value.
        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        # Projection after concatenating all attention heads.
        self.out_proj = nn.Linear(
            d_out,
            d_out,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Create the causal attention mask.
        #
        # Example for sequence length 4:
        #
        # 0 1 1 1
        # 0 0 1 1
        # 0 0 0 1
        # 0 0 0 0
        #
        # Positions containing 1 will be masked.
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length,
                ),
                diagonal=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, num_tokens, _ = (
            x.shape
        )

        
        # Create queries, keys, and values
        

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Current shape:
        #
        # [batch_size, num_tokens, d_out]
        #
        # Split d_out into multiple attention heads.
        keys = keys.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        values = values.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        queries = queries.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        # Move the head dimension before the token dimension.
        #
        # Result:
        #
        # [batch_size, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        
        # Compute attention scores
        

        attention_scores = (
            queries
            @ keys.transpose(2, 3)
        )

        # Select the required part of the causal mask.
        causal_mask = (
            self.mask[
                :num_tokens,
                :num_tokens,
            ].bool()
        )

        # Prevent attention to future tokens.
        attention_scores.masked_fill_(
            causal_mask,
            -torch.inf,
        )

        # Scale the scores before softmax.
        attention_weights = torch.softmax(
            attention_scores
            / (
                self.head_dim ** 0.5
            ),
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        
        # Compute context vectors
        

        context_vectors = (
            attention_weights
            @ values
        )

        # Move tokens before heads again.
        context_vectors = (
            context_vectors
            .transpose(1, 2)
            .contiguous()
        )

        # Combine all attention heads.
        #
        # [B, T, heads, head_dim]
        #
        # becomes
        #
        # [B, T, d_out]
        context_vectors = (
            context_vectors.view(
                batch_size,
                num_tokens,
                self.d_out,
            )
        )

        return self.out_proj(
            context_vectors
        )



# Transformer block


class TransformerBlock(nn.Module):
    """
    One GPT Transformer block.

    Structure:

        LayerNorm
            ↓
        Multi-head attention
            ↓
        Residual connection
            ↓
        LayerNorm
            ↓
        Feed-forward network
            ↓
        Residual connection
    """

    def __init__(
        self,
        cfg: dict,
    ):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=(
                cfg["context_length"]
            ),
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(
            cfg["emb_dim"]
        )

        self.norm2 = LayerNorm(
            cfg["emb_dim"]
        )

        self.drop_shortcut = nn.Dropout(
            cfg["drop_rate"]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

    
        # Attention block + residual connection
        

        shortcut = x

        x = self.norm1(x)

        x = self.att(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        
        # Feed-forward block + residual connection
        

        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        return x



# GPT model


class GPTModel(nn.Module):
    """
    GPT-2 style decoder-only Transformer.

    Input
    -----
    Token IDs:

        [batch_size, sequence_length]

    Output
    ------
    Logits:

        [batch_size, sequence_length, vocab_size]

    Before classification finetuning, vocab_size is 50,257.

    classifier.py will later replace out_head with a
    two-class output layer.
    """

    def __init__(
        self,
        cfg: dict,
    ):
        super().__init__()

        self.cfg = cfg.copy()

        # Token embedding layer.
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
        )

        # Positional embedding layer.
        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"],
        )

        self.drop_emb = nn.Dropout(
            cfg["drop_rate"]
        )

        # Stack of Transformer blocks.
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(cfg)
                for _ in range(
                    cfg["n_layers"]
                )
            ]
        )

        self.final_norm = LayerNorm(
            cfg["emb_dim"]
        )

        # Standard GPT language-model output layer.
        #
        # Later:
        #
        # 768 -> 2
        #
        # for ham/spam classification.
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len = (
            input_ids.shape
        )

        # Protect against sequences that exceed GPT-2's
        # positional embedding table.
        if seq_len > self.cfg[
            "context_length"
        ]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"GPT context length "
                f"{self.cfg['context_length']}."
            )

        
        # Token embeddings
        

        token_embeddings = (
            self.tok_emb(input_ids)
        )

        
        # Positional embeddings
        

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        position_embeddings = (
            self.pos_emb(positions)
        )

        # Broadcasting applies positional embeddings to
        # every sample in the batch.
        x = (
            token_embeddings
            + position_embeddings
        )

    
        # GPT Transformer
        

        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)

        
        # Output logits
    

        logits = self.out_head(x)

        return logits