"""
Verify that the pretrained GPT-2 weights were loaded correctly.

A pretrained GPT-2 model should generate coherent text.

If the architecture or weight mapping is incorrect, generated text
will usually be nonsensical.
"""

import torch

from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)
from spam_detector.model.tokenization import (
    get_gpt2_tokenizer,
)


def generate_greedy(
    model,
    input_ids,
    max_new_tokens: int,
):
    """
    Generate tokens using greedy decoding.

    At every step, select the token with the highest logit.
    """

    for _ in range(
        max_new_tokens
    ):
        # GPT-2 supports at most 1024 input positions.
        input_context = input_ids[
            :,
            -1024:,
        ]

        with torch.no_grad():
            logits = model(
                input_context
            )

        # Only the final token position predicts
        # the next token.
        next_token_logits = (
            logits[
                :,
                -1,
                :,
            ]
        )

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            (
                input_ids,
                next_token,
            ),
            dim=1,
        )

    return input_ids


def main() -> None:
    
    # Load pretrained GPT-2
    

    model = (
        create_pretrained_gpt2_small()
    )

    tokenizer = (
        get_gpt2_tokenizer()
    )

    
    # Prepare a simple prompt
    

    prompt = (
        "Every effort moves you"
    )

    encoded = tokenizer.encode(
        prompt,
        allowed_special={
            "<|endoftext|>"
        },
    )

    input_ids = torch.tensor(
        encoded,
        dtype=torch.long,
    ).unsqueeze(0)

    
    # Generate text
    

    output_ids = generate_greedy(
        model=model,
        input_ids=input_ids,
        max_new_tokens=15,
    )

    generated_text = (
        tokenizer.decode(
            output_ids[
                0
            ].tolist()
        )
    )

    print(
        "\nGenerated text:"
    )

    print(
        generated_text
    )

    
    # Architecture sanity checks
    

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

    assert len(
        model.trf_blocks
    ) == 12

    print(
        "\nPretrained GPT-2 "
        "loaded successfully."
    )


if __name__ == "__main__":
    main()