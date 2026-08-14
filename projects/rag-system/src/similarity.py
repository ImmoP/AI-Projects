"""
This module implements different similarity and distance functions
that can be used to compare embedding vectors.

The mathematical operations are implemented manually to make the
underlying calculations transparent.

Implemented functions:

- dot_product()
- vector_norm()
- cosine_similarity()
- euclidean_distance()
- softmax()
- temperature_softmax()
"""

import math
from collections.abc import Sequence


def validate_vectors(
    vector_a: Sequence[float],
    vector_b: Sequence[float]
) -> None:
    """
    Check whether two vectors can be compared.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same number of dimensions"
        )

    if len(vector_a) == 0:
        raise ValueError(
            "Vectors must not be empty"
        )


def dot_product(
    vector_a: Sequence[float],
    vector_b: Sequence[float]
) -> float:
    """
    Calculate the dot product of two vectors manually.
    """

    validate_vectors(
        vector_a,
        vector_b
    )

    result = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        result += value_a * value_b

    return result


def vector_norm(
    vector: Sequence[float]
) -> float:
    """
    Calculate the Euclidean norm (length) of a vector manually.
    """

    if len(vector) == 0:
        raise ValueError(
            "Vector must not be empty"
        )

    squared_sum = 0.0

    for value in vector:
        squared_sum += value ** 2

    return math.sqrt(squared_sum)


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float]
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    validate_vectors(
        vector_a,
        vector_b
    )

    numerator = dot_product(
        vector_a,
        vector_b
    )

    denominator = (
        vector_norm(vector_a)
        * vector_norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def euclidean_distance(
    vector_a: Sequence[float],
    vector_b: Sequence[float]
) -> float:
    """
    Calculate the Euclidean distance between two vectors manually.
    """

    validate_vectors(
        vector_a,
        vector_b
    )

    squared_distance = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        difference = value_a - value_b
        squared_distance += difference ** 2

    return math.sqrt(squared_distance)


def softmax(
    scores: Sequence[float]
) -> list[float]:
    """
    Convert a list of scores into a normalized distribution
    using numerically stable softmax.
    """

    if len(scores) == 0:
        return []

    max_score = max(scores)

    exponentials = []

    for score in scores:
        exponentials.append(
            math.exp(score - max_score)
        )

    exponential_sum = sum(exponentials)

    probabilities = []

    for value in exponentials:
        probabilities.append(
            value / exponential_sum
        )

    return probabilities


def temperature_softmax(
    scores: Sequence[float],
    temperature: float = 1.0
) -> list[float]:
    """
    Apply softmax with temperature scaling.

    Lower temperature produces a sharper distribution.
    Higher temperature produces a flatter distribution.
    """

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than 0"
        )

    scaled_scores = []

    for score in scores:
        scaled_scores.append(
            score / temperature
        )

    return softmax(
        scaled_scores
    )


if __name__ == "__main__":

    vector_a = [1.0, 2.0, 3.0]
    vector_b = [2.0, 4.0, 6.0]
    vector_c = [3.0, -1.0, 0.5]

    print("Vector A:", vector_a)
    print("Vector B:", vector_b)
    print("Vector C:", vector_c)

    print()
    print("A vs B")
    print(f"Dot product:        {dot_product(vector_a, vector_b):.4f}")
    print(f"Cosine similarity:  {cosine_similarity(vector_a, vector_b):.4f}")
    print(f"Euclidean distance: {euclidean_distance(vector_a, vector_b):.4f}")

    print()
    print("A vs C")
    print(f"Dot product:        {dot_product(vector_a, vector_c):.4f}")
    print(f"Cosine similarity:  {cosine_similarity(vector_a, vector_c):.4f}")
    print(f"Euclidean distance: {euclidean_distance(vector_a, vector_c):.4f}")

    scores = [
        0.82,
        0.78,
        0.70,
        0.45
    ]

    print()
    print("Similarity scores:")
    print(scores)

    print()
    print("Softmax:")
    print(softmax(scores))

    print()
    print("Temperature softmax T=1.0:")
    print(temperature_softmax(scores, temperature=1.0))

    print()
    print("Temperature softmax T=0.1:")
    print(temperature_softmax(scores, temperature=0.1))

    print()
    print("Temperature softmax T=2.0:")
    print(temperature_softmax(scores, temperature=2.0))