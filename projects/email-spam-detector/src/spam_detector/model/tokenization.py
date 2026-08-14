"""
Tokenization utilities for the spam classification model.

The tokenizer converts a formatted email into GPT-2 token IDs.

Goals:
- Keep sender and subject whenever possible.
- Limit the total sequence length to the model context length.
- For long email bodies, keep both the beginning and the end.
- Append an end-of-text token to every email.
- Padding will be handled later when
  multiple emails are combined into a batch.

Tokenization flow:

sender + subject + body
        ↓
format_email()
        ↓
GPT-2 tokenizer
        ↓
if sequence is too long:
    keep metadata
    + beginning of body
    + end of body
        ↓
append <|endoftext|>
        ↓
input_ids
"""

from dataclasses import dataclass

import tiktoken

from spam_detector.model.text_formatting import format_email

# Maximum sequence length supported by GPT-2.
DEFAULT_CONTEXT_LENGTH = 1024

# Percentage of the available body tokens taken from the beginning.
BODY_HEAD_RATIO = 0.75

# GPT-2 uses token ID 50256 for <|endoftext|>.
GPT2_EOS_TOKEN_ID = 50256


@dataclass
class TokenizedEmail:
    """
    Result of tokenizing one email.

    Attributes
    ----------
    input_ids:
        Final token IDs that will later be passed to the model.

    original_token_count:
        Number of tokens before truncation.

    final_token_count:
        Number of tokens after truncation and after adding EOS.

    was_truncated:
        True if information had to be removed.

    metadata_truncated:
        True if sender/subject alone exceeded the available
        context length. This should be extremely rare.
    """

    input_ids: list[int]
    original_token_count: int
    final_token_count: int
    was_truncated: bool
    metadata_truncated: bool


def encode_text(
    text: str,
    tokenizer: tiktoken.Encoding,
) -> list[int]:
    """
    Encode text using the GPT-2 tokenizer.

    disallowed_special=() ensures that strings resembling special
    tokens inside an email are treated as normal text.
    """
    return tokenizer.encode(
        text,
        disallowed_special=(),
    )


def truncate_head_tail(
    token_ids: list[int],
    max_tokens: int,
    head_ratio: float = BODY_HEAD_RATIO,
) -> list[int]:
    """
    Truncate a token sequence while keeping both its beginning
    and its end.

    Example
    -------
    If 400 body tokens are available and head_ratio = 0.75:

        300 tokens are taken from the beginning
        100 tokens are taken from the end

    This is useful for emails because important information may
    occur both near the beginning and near the end.
    """
    if len(token_ids) <= max_tokens:
        return token_ids

    if max_tokens <= 0:
        return []

    head_tokens = int(max_tokens * head_ratio)
    tail_tokens = max_tokens - head_tokens

    # If no tail tokens are available, only keep the beginning.
    if tail_tokens == 0:
        return token_ids[:head_tokens]

    return (
        token_ids[:head_tokens]
        + token_ids[-tail_tokens:]
    )


def tokenize_email(
    sender,
    subject,
    text,
    tokenizer: tiktoken.Encoding,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> TokenizedEmail:
    """
    Tokenize one email for the GPT-2 classifier.

    Sender and subject are treated as metadata and are protected
    from truncation whenever possible.

    If the complete email is longer than context_length, the body
    is shortened using a head-tail strategy.

    One token position is reserved for the GPT-2 end-of-text token.
    """
    if context_length < 2:
        raise ValueError(
            "context_length must be at least 2."
        )

    # Reserve one position for the EOS token.
    content_budget = context_length - 1

    # Build the complete formatted email.
    full_text = format_email(
        sender,
        subject,
        text,
    )

    # Encode the complete email once so that we know its true
    # original token length.
    full_token_ids = encode_text(
        full_text,
        tokenizer,
    )

    original_token_count = len(full_token_ids)

    # ----------------------------------------------------------
    # Case 1:
    # The complete email already fits into the context window.
    # ----------------------------------------------------------
    if original_token_count <= content_budget:
        final_ids = (
            full_token_ids
            + [GPT2_EOS_TOKEN_ID]
        )

        return TokenizedEmail(
            input_ids=final_ids,
            original_token_count=original_token_count,
            final_token_count=len(final_ids),
            was_truncated=False,
            metadata_truncated=False,
        )

    # ----------------------------------------------------------
    # Case 2:
    # The email is too long.
    #
    # Sender and subject are formatted separately from the body
    # so that metadata can be preserved.
    # ----------------------------------------------------------

    metadata_text = format_email(
        sender,
        subject,
        "",
    )

    body_text = format_email(
        "",
        "",
        text,
    )

    metadata_ids = encode_text(
        metadata_text,
        tokenizer,
    )

    body_ids = encode_text(
        body_text,
        tokenizer,
    )

    # The normal email format contains an empty line between the
    # metadata section and the body.
    if metadata_text and body_text:
        separator_ids = encode_text(
            "\n\n",
            tokenizer,
        )
    else:
        separator_ids = []

    metadata_truncated = False

    # ----------------------------------------------------------
    # Extremely unusual case:
    # Sender + subject alone are longer than the available
    # context window.
    # ----------------------------------------------------------
    if len(metadata_ids) >= content_budget:
        content_ids = metadata_ids[
            :content_budget
        ]

        metadata_truncated = True

    else:
        # Calculate how much room remains for the body.
        remaining_body_tokens = (
            content_budget
            - len(metadata_ids)
            - len(separator_ids)
        )

        # If the separator itself would exceed the remaining
        # capacity, omit it.
        if remaining_body_tokens < 0:
            separator_ids = []

            remaining_body_tokens = (
                content_budget
                - len(metadata_ids)
            )

        truncated_body_ids = truncate_head_tail(
            body_ids,
            max_tokens=remaining_body_tokens,
            head_ratio=BODY_HEAD_RATIO,
        )

        content_ids = (
            metadata_ids
            + separator_ids
            + truncated_body_ids
        )

    # Final safety check.
    content_ids = content_ids[:content_budget]

    # Append EOS so that every email has a clear final
    # classification position.
    final_ids = (
        content_ids
        + [GPT2_EOS_TOKEN_ID]
    )

    return TokenizedEmail(
        input_ids=final_ids,
        original_token_count=original_token_count,
        final_token_count=len(final_ids),
        was_truncated=True,
        metadata_truncated=metadata_truncated,
    )


def get_gpt2_tokenizer() -> tiktoken.Encoding:
    """
    Return the GPT-2 tokenizer used by the classification model.
    """
    return tiktoken.get_encoding("gpt2")