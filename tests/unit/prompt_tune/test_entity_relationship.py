# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for generate_entity_relationship_examples."""

from typing import TYPE_CHECKING, Any, Unpack

from graphrag.prompt_tune.generator.entity_relationship import (
    generate_entity_relationship_examples,
)
from graphrag_llm.completion import LLMCompletion
from graphrag_llm.utils import create_completion_response

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from graphrag_llm.metrics import MetricsStore
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        LLMCompletionArgs,
        LLMCompletionChunk,
        LLMCompletionResponse,
        ResponseFormat,
    )

PERSONA = "You are a helpful assistant."
DOCS = [f"document number {n}" for n in range(1, 6)]


class RecordingCompletion(LLMCompletion):
    """A model that records the messages of every call it receives.

    MockLLMCompletion drops `messages` on the floor, so it cannot be used to assert
    anything about the payload.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.calls: list[Any] = []

    def completion(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | Iterator[LLMCompletionChunk]":
        self.calls.append(kwargs["messages"])
        return create_completion_response(f"example {len(self.calls)}")  # type: ignore

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


async def _generate(model: RecordingCompletion, docs: list[str]) -> list[str]:
    _, examples = await generate_entity_relationship_examples(
        model,
        persona=PERSONA,
        entity_types=["organization", "person"],
        docs=docs,
        language="English",
        max_token_count=1_000_000,
    )
    return examples


async def test_each_call_gets_only_its_own_document():
    model = RecordingCompletion()

    await _generate(model, DOCS)

    assert len(model.calls) == len(DOCS)
    for call, doc in zip(model.calls, DOCS, strict=True):
        assert [message["role"] for message in call] == ["system", "user"]
        assert call[0]["content"] == PERSONA
        assert doc in call[1]["content"]


async def test_calls_do_not_share_a_message_list():
    model = RecordingCompletion()

    await _generate(model, DOCS)

    assert len({id(call) for call in model.calls}) == len(DOCS)


async def test_no_cross_document_leakage():
    model = RecordingCompletion()

    await _generate(model, DOCS)

    for index, call in enumerate(model.calls):
        rendered = "".join(str(message["content"]) for message in call)
        for other in DOCS[:index] + DOCS[index + 1 :]:
            assert other not in rendered


async def test_examples_returned_in_document_order():
    model = RecordingCompletion()

    examples = await _generate(model, DOCS)

    assert examples == [f"example {n}" for n in range(1, len(DOCS) + 1)]
