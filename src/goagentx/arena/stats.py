"""Statistical tests for Arena paired evaluations."""

from __future__ import annotations

import itertools
import random
from enum import StrEnum
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.arena.paired_eval import PairedEvaluationResult


DEFAULT_MIN_SAMPLES = 5
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_EXACT_MAX_SAMPLES = 16
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0
EPSILON = 1e-12


class SignificanceTestError(RuntimeError):
    """Raised when a significance test cannot be configured."""


class PermutationAlternative(StrEnum):
    """Alternative hypotheses for paired permutation tests."""

    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two_sided"


class StrictModel(BaseModel):
    """Base model that rejects unknown statistical result fields."""

    model_config = ConfigDict(extra="forbid")


class SignificanceResult(StrictModel):
    """Permutation-test result for paired score deltas."""

    method: str = "paired_permutation"
    alternative: PermutationAlternative
    sample_count: int = Field(..., ge=0)
    effective_sample_count: int = Field(..., ge=0)
    observed_mean_delta: float
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    alpha: float = Field(..., ge=0.0, le=1.0)
    is_significant: bool
    insufficient_sample: bool
    permutations: int = Field(..., ge=0)
    seed: int | None = None
    reason: str | None = None


def permutation_test_score_deltas(
    score_deltas: Sequence[float],
    *,
    alternative: PermutationAlternative | str = PermutationAlternative.GREATER,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
    exact_max_samples: int = DEFAULT_EXACT_MAX_SAMPLES,
    seed: int | None = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> SignificanceResult:
    """Run a paired sign-flip permutation test on candidate score deltas."""
    deltas = [float(delta) for delta in score_deltas]
    _validate_test_config(
        deltas=deltas,
        min_samples=min_samples,
        permutations=permutations,
        exact_max_samples=exact_max_samples,
        alpha=alpha,
    )

    resolved_alternative = PermutationAlternative(alternative)
    observed_mean_delta = _mean(deltas)
    non_zero_deltas = [delta for delta in deltas if abs(delta) > EPSILON]
    if len(non_zero_deltas) < min_samples:
        return SignificanceResult(
            alternative=resolved_alternative,
            sample_count=len(deltas),
            effective_sample_count=len(non_zero_deltas),
            observed_mean_delta=observed_mean_delta,
            p_value=None,
            alpha=alpha,
            is_significant=False,
            insufficient_sample=True,
            permutations=0,
            seed=seed,
            reason=(
                f"need at least {min_samples} non-zero paired deltas, "
                f"got {len(non_zero_deltas)}"
            ),
        )

    is_exact = len(non_zero_deltas) <= exact_max_samples
    permuted_means = _permuted_means(
        deltas=deltas,
        permutations=permutations,
        exact_max_samples=exact_max_samples,
        seed=seed,
    )
    extreme_count = sum(
        1
        for permuted_mean in permuted_means
        if _is_extreme(
            permuted_mean=permuted_mean,
            observed_mean=observed_mean_delta,
            alternative=resolved_alternative,
        )
    )
    p_value = (
        extreme_count / len(permuted_means)
        if is_exact
        else (extreme_count + 1) / (len(permuted_means) + 1)
    )

    return SignificanceResult(
        alternative=resolved_alternative,
        sample_count=len(deltas),
        effective_sample_count=len(non_zero_deltas),
        observed_mean_delta=observed_mean_delta,
        p_value=p_value,
        alpha=alpha,
        is_significant=p_value <= alpha,
        insufficient_sample=False,
        permutations=len(permuted_means),
        seed=seed,
    )


def permutation_test_paired_result(
    evaluation: PairedEvaluationResult,
    **kwargs: object,
) -> SignificanceResult:
    """Run the default paired permutation test for a D1 evaluation result."""
    return permutation_test_score_deltas(
        [result.score_delta for result in evaluation.results],
        **kwargs,
    )


def _validate_test_config(
    *,
    deltas: Sequence[float],
    min_samples: int,
    permutations: int,
    exact_max_samples: int,
    alpha: float,
) -> None:
    """Reject invalid permutation-test inputs."""
    if not deltas:
        raise SignificanceTestError("permutation test requires at least one score delta")
    if min_samples <= 0:
        raise SignificanceTestError("min_samples must be greater than 0")
    if permutations <= 0:
        raise SignificanceTestError("permutations must be greater than 0")
    if exact_max_samples < 0:
        raise SignificanceTestError("exact_max_samples must be greater than or equal to 0")
    if not 0 <= alpha <= 1:
        raise SignificanceTestError("alpha must be between 0 and 1")


def _permuted_means(
    *,
    deltas: Sequence[float],
    permutations: int,
    exact_max_samples: int,
    seed: int | None,
) -> list[float]:
    """Return exact or sampled sign-flip permutation means."""
    non_zero_deltas = [delta for delta in deltas if abs(delta) > EPSILON]
    denominator = len(deltas)

    if len(non_zero_deltas) <= exact_max_samples:
        return [
            sum(
                sign * delta
                for sign, delta in zip(signs, non_zero_deltas, strict=True)
            )
            / denominator
            for signs in itertools.product((-1, 1), repeat=len(non_zero_deltas))
        ]

    rng = random.Random(seed)
    return [
        sum(rng.choice((-1, 1)) * delta for delta in non_zero_deltas) / denominator
        for _ in range(permutations)
    ]


def _is_extreme(
    *,
    permuted_mean: float,
    observed_mean: float,
    alternative: PermutationAlternative,
) -> bool:
    """Return whether a permuted mean is at least as extreme as observed."""
    if alternative is PermutationAlternative.GREATER:
        return permuted_mean >= observed_mean - EPSILON
    if alternative is PermutationAlternative.LESS:
        return permuted_mean <= observed_mean + EPSILON
    return abs(permuted_mean) >= abs(observed_mean) - EPSILON


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean for a non-empty sequence."""
    return sum(values) / len(values)
