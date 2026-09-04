# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for create_extract_graph_prompt."""

import logging
import re
from typing import Any

import pytest
from graphrag.prompt_tune.generator.extract_graph_prompt import (
    count_base_prompt_tokens,
    create_extract_graph_prompt,
)
from graphrag.prompt_tune.template.extract_graph import (
    EXAMPLE_EXTRACTION_TEMPLATE,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
)
from graphrag_llm.tokenizer import Tokenizer

ENTITY_TYPES = "organization, person"
LANGUAGE = "English"
# braces arrive doubled from load_docs_in_chunks and must survive untouched
DOCS = ["document " + str(n) + " with {{braces}} in it" for n in range(5)]
EXAMPLES = ["output of document " + str(n) for n in range(5)]


class CharTokenizer(Tokenizer):
    """One character, one token, so budgets in these tests are readable."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _example_cost(index: int, entity_types: str | None) -> int:
    template = (
        EXAMPLE_EXTRACTION_TEMPLATE
        if entity_types
        else UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE
    )
    arguments: dict[str, Any] = {
        "n": index + 1,
        "input_text": DOCS[index],
        "output": EXAMPLES[index],
    }
    if entity_types:
        arguments["entity_types"] = entity_types
    return CharTokenizer().num_tokens(template.format(**arguments))


def _budget_for(count: int, entity_types: str | None) -> int:
    """The smallest budget that fits exactly `count` examples."""
    base = count_base_prompt_tokens(entity_types, LANGUAGE, CharTokenizer())
    return base + sum(_example_cost(i, entity_types) for i in range(count))


def _build(entity_types: str | None, max_token_count: int, **kwargs: Any) -> str:
    return create_extract_graph_prompt(
        entity_types=entity_types,
        docs=DOCS,
        examples=EXAMPLES,
        language=LANGUAGE,
        max_token_count=max_token_count,
        tokenizer=CharTokenizer(),
        **kwargs,
    )


def _example_blocks(prompt: str) -> dict[int, str]:
    parts = re.split(r"^Example (\d+):$", prompt, flags=re.MULTILINE)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


@pytest.mark.parametrize("entity_types", [ENTITY_TYPES, None])
def test_base_token_count_matches_prompt_with_no_examples(entity_types):
    tokenizer = CharTokenizer()

    empty = create_extract_graph_prompt(
        entity_types=entity_types,
        docs=[],
        examples=[],
        language=LANGUAGE,
        max_token_count=0,
        tokenizer=tokenizer,
    )

    assert count_base_prompt_tokens(entity_types, LANGUAGE, tokenizer) == (
        tokenizer.num_tokens(empty)
    )


def test_keeps_every_example_when_budget_is_ample():
    prompt = _build(ENTITY_TYPES, _budget_for(len(EXAMPLES), ENTITY_TYPES))

    assert sorted(_example_blocks(prompt)) == [1, 2, 3, 4, 5]


def test_drops_examples_beyond_the_budget():
    prompt = _build(ENTITY_TYPES, _budget_for(3, ENTITY_TYPES))

    assert sorted(_example_blocks(prompt)) == [1, 2, 3]


def test_keeps_minimum_examples_even_with_zero_budget():
    prompt = _build(ENTITY_TYPES, 0, min_examples_required=2)

    assert sorted(_example_blocks(prompt)) == [1, 2]


def test_untyped_mode_honours_the_budget():
    prompt = _build(None, _budget_for(len(EXAMPLES), None))

    assert sorted(_example_blocks(prompt)) == [1, 2, 3, 4, 5]


def test_untyped_mode_drops_examples_beyond_the_budget():
    prompt = _build(None, _budget_for(3, None))

    assert sorted(_example_blocks(prompt)) == [1, 2, 3]


@pytest.mark.parametrize("entity_types", [ENTITY_TYPES, None])
def test_each_example_carries_its_own_document(entity_types):
    blocks = _example_blocks(_build(entity_types, _budget_for(5, entity_types)))

    for number, block in blocks.items():
        assert DOCS[number - 1] in block


def test_warns_with_the_required_budget_when_the_minimum_overruns(caplog):
    with caplog.at_level(logging.WARNING):
        _build(ENTITY_TYPES, 0, min_examples_required=2)

    assert f"--max-tokens {_budget_for(len(EXAMPLES), ENTITY_TYPES)}" in caplog.text
