# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the example budget of generate_entity_relationship_examples."""

import asyncio
from typing import TYPE_CHECKING, Any, Unpack

import pytest
from graphrag.prompt_tune.generator.entity_relationship import (
    MAX_EXAMPLES,
    generate_entity_relationship_examples,
)
from graphrag.prompt_tune.generator.extract_graph_prompt import (
    count_base_prompt_tokens,
    format_extract_graph_example,
)
from graphrag_llm.completion import LLMCompletion
from graphrag_llm.tokenizer import Tokenizer
from graphrag_llm.utils import create_completion_response

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from graphrag_llm.metrics import MetricsStore
    from graphrag_llm.types import (
        LLMCompletionArgs,
        LLMCompletionChunk,
        LLMCompletionResponse,
        ResponseFormat,
    )

PERSONA = "You are a helpful assistant."
ENTITY_TYPES = "organization, person"
LANGUAGE = "English"
# equal length, so every example costs the same and budgets stay readable
DOCS = ["document " + chr(ord("A") + n) for n in range(10)]
RESPONSES = ["example " + chr(ord("a") + n) for n in range(10)]


class CharTokenizer(Tokenizer):
    """One character, one token."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class RecordingCompletion(LLMCompletion):
    """A model that records every call and answers from a fixed script."""

    def __init__(self, responses: list[str] | None = None, **kwargs: Any) -> None:
        self._responses = responses if responses is not None else RESPONSES
        self.calls: list[Any] = []

    def completion(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | Iterator[LLMCompletionChunk]":
        index = len(self.calls)
        self.calls.append(kwargs["messages"])
        return create_completion_response(self._responses[index])  # type: ignore

    async def completion_async(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | AsyncIterator[LLMCompletionChunk]":
        return self.completion(**kwargs)  # type: ignore

    @property
    def metrics_store(self) -> "MetricsStore":
        raise NotImplementedError

    @property
    def tokenizer(self) -> "Tokenizer":
        raise NotImplementedError


class GatedCompletion(RecordingCompletion):
    """Answers only once `expected` calls are in flight at the same time."""

    def __init__(self, expected: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._expected = expected
        self._in_flight = 0
        self._reached = asyncio.Event()

    async def completion_async(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | AsyncIterator[LLMCompletionChunk]":
        self._in_flight += 1
        if self._in_flight >= self._expected:
            self._reached.set()
        await self._reached.wait()
        return self.completion(**kwargs)  # type: ignore


def _example_cost(index: int) -> int:
    return CharTokenizer().num_tokens(
        format_extract_graph_example(
            n=index + 1,
            input_text=DOCS[index],
            output=RESPONSES[index],
            entity_types=ENTITY_TYPES,
        )
    )


def _budget_for(count: int) -> int:
    """The smallest budget that fits exactly `count` examples."""
    base = count_base_prompt_tokens(ENTITY_TYPES, LANGUAGE, CharTokenizer())
    return base + sum(_example_cost(i) for i in range(count))


async def _generate(
    model: RecordingCompletion,
    max_token_count: int,
    docs: list[str] | None = None,
    **kwargs: Any,
) -> tuple[list[str], list[str]]:
    return await generate_entity_relationship_examples(
        model,
        persona=PERSONA,
        entity_types=ENTITY_TYPES,
        docs=DOCS[:5] if docs is None else docs,
        language=LANGUAGE,
        tokenizer=CharTokenizer(),
        max_token_count=max_token_count,
        **kwargs,
    )


async def test_stops_calling_once_the_budget_is_spent():
    model = RecordingCompletion()

    _, examples = await _generate(model, _budget_for(3))

    assert len(model.calls) == 3
    assert len(examples) == 3


async def test_generates_the_required_minimum_with_zero_budget():
    model = RecordingCompletion()

    _, examples = await _generate(model, 0, min_examples_required=2)

    assert len(model.calls) == 2
    assert len(examples) == 2


async def test_never_exceeds_max_examples_with_an_unlimited_budget():
    model = RecordingCompletion()

    _, examples = await _generate(model, 1_000_000, docs=DOCS)

    assert len(model.calls) == MAX_EXAMPLES
    assert len(examples) == MAX_EXAMPLES


async def test_returns_docs_and_examples_of_equal_length():
    model = RecordingCompletion()

    docs_used, examples = await _generate(model, _budget_for(3))

    assert len(docs_used) == len(examples)
    assert docs_used == DOCS[: len(examples)]


async def test_discards_an_example_that_overruns_after_generation():
    responses = list(RESPONSES)
    responses[3] = "x" * 10_000
    model = RecordingCompletion(responses)

    # room for the three cheap examples and two thirds of a fourth: the average says
    # go ahead, the answer that comes back does not fit
    _, examples = await _generate(model, _budget_for(3) + 2 * _example_cost(3))

    assert len(model.calls) == 4
    assert len(examples) == 3


async def test_does_not_call_when_the_average_example_cannot_fit():
    model = RecordingCompletion()

    _, examples = await _generate(model, _budget_for(2) + _example_cost(2) - 1)

    assert len(model.calls) == 2
    assert len(examples) == 2


async def test_min_examples_above_max_examples_still_generates_the_minimum():
    model = RecordingCompletion()

    _, examples = await _generate(model, 0, docs=DOCS, min_examples_required=8)

    assert len(model.calls) == 8
    assert len(examples) == 8


async def test_required_examples_are_issued_concurrently():
    model = GatedCompletion(expected=3)

    _, examples = await asyncio.wait_for(
        _generate(model, 0, min_examples_required=3), timeout=10
    )

    assert len(model.calls) == 3
    assert len(examples) == 3


@pytest.mark.parametrize("min_examples_required", [1, 2, 3])
async def test_examples_stay_aligned_with_their_documents(min_examples_required):
    model = RecordingCompletion()

    docs_used, examples = await _generate(
        model, 1_000_000, min_examples_required=min_examples_required
    )

    for doc, call in zip(docs_used, model.calls, strict=True):
        assert doc in call[1]["content"]
    assert examples == RESPONSES[: len(examples)]
