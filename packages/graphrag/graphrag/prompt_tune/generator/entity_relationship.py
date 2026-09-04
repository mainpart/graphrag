# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity relationship example generation module."""

import asyncio
import logging
from typing import TYPE_CHECKING

from graphrag_llm.utils import (
    CompletionMessagesBuilder,
)

from graphrag.prompt_tune.defaults import MAX_TOKEN_COUNT
from graphrag.prompt_tune.generator.extract_graph_prompt import (
    count_base_prompt_tokens,
    format_extract_graph_example,
)
from graphrag.prompt_tune.prompt.entity_relationship import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
)
from graphrag.tokenizer.get_tokenizer import get_tokenizer

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

MAX_EXAMPLES = 5


async def _generate_example(
    model: "LLMCompletion",
    persona: str,
    message: str,
    json_mode: bool,
) -> str:
    """Generate a single entity/relationship example."""
    response = await model.completion_async(
        messages=CompletionMessagesBuilder()
        .add_system_message(persona)
        .add_user_message(message)
        .build(),
        response_format_json_object=json_mode,
    )
    return response.content  # type: ignore


async def generate_entity_relationship_examples(
    model: "LLMCompletion",
    persona: str,
    entity_types: str | list[str] | None,
    docs: str | list[str],
    language: str,
    json_mode: bool = False,
    tokenizer: "Tokenizer | None" = None,
    max_token_count: int = MAX_TOKEN_COUNT,
    min_examples_required: int = 2,
) -> tuple[list[str], list[str]]:
    """Generate entity/relationship examples for use in generating an entity configuration.

    Only as many examples are generated as the extract graph prompt has room for. The first
    `min_examples_required` are generated concurrently and unconditionally; each further one
    is generated only while the remaining budget can be expected to hold it, and is dropped
    again if it turns out not to.

    Will return entity/relationships examples as either JSON or in tuple_delimiter format
    depending on the json_mode parameter.

    Returns
    -------
    - tuple[list[str], list[str]]: the documents the examples were generated from, and the
      examples themselves, aligned position by position
    """
    docs_list = [docs] if isinstance(docs, str) else docs

    if isinstance(entity_types, list):
        entity_types = ", ".join(map(str, entity_types))

    if entity_types:
        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(entity_types=entity_types, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        messages = [
            UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                input_text=doc, language=language
            )
            for doc in docs_list
        ]

    messages = messages[: max(MAX_EXAMPLES, min_examples_required)]

    tokenizer = tokenizer or get_tokenizer()
    tokens_left = max_token_count - count_base_prompt_tokens(
        entity_types, language, tokenizer, json_mode
    )

    def example_cost(index: int, example: str) -> int:
        return tokenizer.num_tokens(
            format_extract_graph_example(
                n=index + 1,
                input_text=docs_list[index],
                output=example,
                entity_types=entity_types,
            )
        )

    required = min(min_examples_required, len(messages))
    examples = list(
        await asyncio.gather(*[
            _generate_example(model, persona, message, json_mode)
            for message in messages[:required]
        ])
    )
    example_tokens = [example_cost(i, example) for i, example in enumerate(examples)]
    tokens_left -= sum(example_tokens)

    # Every further example costs a request, so ask for one only while the budget can be
    # expected to hold it. The estimate is the average of what the earlier ones cost.
    for index in range(required, len(messages)):
        if (
            not example_tokens
            or sum(example_tokens) / len(example_tokens) > tokens_left
        ):
            break

        example = await _generate_example(model, persona, messages[index], json_mode)
        cost = example_cost(index, example)
        if cost > tokens_left:
            break

        examples.append(example)
        example_tokens.append(cost)
        tokens_left -= cost

    logger.info(
        "Generated %d of %d candidate entity relationship examples, %d tokens of the"
        " %d token budget left.",
        len(examples),
        len(messages),
        tokens_left,
        max_token_count,
    )

    return docs_list[: len(examples)], examples
