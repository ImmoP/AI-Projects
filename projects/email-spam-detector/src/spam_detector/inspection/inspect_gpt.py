"""
Inspect the GPT-2 model architecture before loading pretrained weights.

This test only verifies:

- model construction;
- parameter shapes;
- GPT-2 configuration;
- one small forward pass.

The model still contains randomly initialized weights.
"""

import torch

from spam_detector.model.gpt import (
    GPT2_SMALL_CONFIG,
    GPTModel,
)


def main() -> None:

    # Make random initialization reproducible.
    torch.manual_seed(123)

    
    # Create GPT-2 small
    

    model = GPTModel(
        GPT2_SMALL_CONFIG
    )

    print("GPT-2 configuration:")

    for key, value in (
        GPT2_SMALL_CONFIG.items()
    ):
        print(
            f"{key}: {value}"
        )

    
    # Check important architecture properties
    

    print("\n" + "=" * 70)

    print(
        "Token embedding shape:",
        model.tok_emb.weight.shape,
    )

    print(
        "Position embedding shape:",
        model.pos_emb.weight.shape,
    )

    print(
        "Number of Transformer blocks:",
        len(model.trf_blocks),
    )

    print(
        "Output head:",
        model.out_head,
    )

    print(
        "Number of attention heads:",
        model.trf_blocks[
            0
        ].att.num_heads,
    )

    print(
        "Attention head dimension:",
        model.trf_blocks[
            0
        ].att.head_dim,
    )

    
    # Parameter count
    

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}",
    )

    
    # Small forward-pass test
    
    
    # Do NOT test a full [8, 1024] batch yet.
    #
    # The weights are still random and a full sequence would
    # unnecessarily require substantial memory.

    input_ids = torch.tensor(
        [
            [
                15496,
                11,
                314,
                716,
            ],
            [
                40,
                1101,
                257,
                1332,
            ],
        ],
        dtype=torch.long,
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            input_ids
        )

    print(
        "\nInput shape:",
        input_ids.shape,
    )

    print(
        "Output shape:",
        logits.shape,
    )

    
    # Sanity checks
    

    assert (
        model.tok_emb.weight.shape
        == torch.Size(
            [50_257, 768]
        )
    )

    assert (
        model.pos_emb.weight.shape
        == torch.Size(
            [1_024, 768]
        )
    )

    assert (
        len(model.trf_blocks)
        == 12
    )

    assert (
        model.trf_blocks[
            0
        ].att.num_heads
        == 12
    )

    assert (
        model.trf_blocks[
            0
        ].att.head_dim
        == 64
    )

    assert (
        logits.shape
        == torch.Size(
            [2, 4, 50_257]
        )
    )

    print("\n" + "=" * 70)

    print(
        "All GPT architecture checks "
        "passed successfully."
    )


if __name__ == "__main__":
    main()