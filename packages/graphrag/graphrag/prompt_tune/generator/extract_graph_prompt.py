# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity Extraction prompt generator module."""

import logging
from pathlib import Path

from graphrag_llm.tokenizer import Tokenizer

from graphrag.prompt_tune.template.extract_graph import (
    EXAMPLE_EXTRACTION_TEMPLATE,
    GRAPH_EXTRACTION_JSON_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
    UNTYPED_GRAPH_EXTRACTION_PROMPT,
)
from graphrag.tokenizer.get_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

EXTRACT_GRAPH_FILENAME = "extract_graph.txt"


def _select_template(entity_types: str | None, json_mode: bool) -> str:
    """Select the extract graph template for the mode the prompt is built for."""
    if not entity_types:
        return UNTYPED_GRAPH_EXTRACTION_PROMPT
    return GRAPH_EXTRACTION_JSON_PROMPT if json_mode else GRAPH_EXTRACTION_PROMPT


def count_base_prompt_tokens(
    entity_types: str | None,
    language: str,
    tokenizer: Tokenizer,
    json_mode: bool = False,
) -> int:
    """Count the tokens of the extract graph prompt before any example is added.

    The template is formatted with an empty example block, so placeholders are not
    counted and every occurrence of the entity types is.

    Parameters
    ----------
    - entity_types (str | None): The entity types to extract, already joined
    - language (str): The language of the inputs and outputs
    - tokenizer (Tokenizer): The tokenizer to use for encoding text
    - json_mode (bool): Whether to use JSON mode for the prompt. Default is False

    Returns
    -------
    - int: The number of tokens taken by the prompt without examples
    """
    template = _select_template(entity_types, json_mode)
    base_prompt = (
        template.format(entity_types=entity_types, examples="", language=language)
        if entity_types
        else template.format(examples="", language=language)
    )
    return tokenizer.num_tokens(base_prompt)


def format_extract_graph_example(
    n: int,
    input_text: str,
    output: str,
    entity_types: str | None,
) -> str:
    """Format a single example block of the extract graph prompt.

    Parameters
    ----------
    - n (int): The 1-based number of the example
    - input_text (str): The document the example was generated from
    - output (str): The entities and relationships extracted from it
    - entity_types (str | None): The entity types to extract, already joined

    Returns
    -------
    - str: The formatted example block
    """
    if entity_types:
        return EXAMPLE_EXTRACTION_TEMPLATE.format(
            n=n, input_text=input_text, entity_types=entity_types, output=output
        )
    return UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE.format(
        n=n, input_text=input_text, output=output
    )


def select_examples_within_budget(
    docs: list[str],
    examples: list[str],
    entity_types: str | None,
    base_token_count: int,
    max_token_count: int,
    min_examples_required: int,
    tokenizer: Tokenizer,
) -> tuple[str, int, int]:
    """Format as many examples as the token budget admits.

    The first `min_examples_required` examples are added whether they fit or not, so the
    prompt is never left without examples.

    Parameters
    ----------
    - docs (list[str]): The documents the examples were generated from
    - examples (list[str]): The generated examples, aligned with `docs`
    - entity_types (str | None): The entity types to extract, already joined
    - base_token_count (int): The size of the prompt before any example is added
    - max_token_count (int): The maximum number of tokens to use for the prompt
    - min_examples_required (int): The number of examples added unconditionally
    - tokenizer (Tokenizer): The tokenizer to use for encoding text

    Returns
    -------
    - tuple[str, int, int]: the formatted example block, how many examples went into it,
      and the `max_token_count` at which every supplied example would have fitted
    """
    formatted = [
        format_extract_graph_example(
            n=i + 1, input_text=docs[i], output=output, entity_types=entity_types
        )
        for i, output in enumerate(examples)
    ]
    example_tokens = [tokenizer.num_tokens(block) for block in formatted]

    tokens_left = max_token_count - base_token_count
    examples_prompt = ""
    examples_used = 0

    for i, block in enumerate(formatted):
        # the first min_examples_required go in whether they fit or not
        if i >= min_examples_required and example_tokens[i] > tokens_left:
            break

        examples_prompt += block
        examples_used += 1
        tokens_left -= example_tokens[i]

    return examples_prompt, examples_used, base_token_count + sum(example_tokens)


def create_extract_graph_prompt(
    entity_types: str | list[str] | None,
    docs: list[str],
    examples: list[str],
    language: str,
    max_token_count: int,
    tokenizer: Tokenizer | None = None,
    json_mode: bool = False,
    output_path: Path | None = None,
    min_examples_required: int = 2,
) -> str:
    """
    Create a prompt for entity extraction.

    Parameters
    ----------
    - entity_types (str | list[str]): The entity types to extract
    - docs (list[str]): The list of documents to extract entities from
    - examples (list[str]): The list of examples to use for entity extraction
    - language (str): The language of the inputs and outputs
    - tokenizer (Tokenizer): The tokenizer to use for encoding and decoding text.
    - max_token_count (int): The maximum number of tokens to use for the prompt
    - json_mode (bool): Whether to use JSON mode for the prompt. Default is False
    - output_path (Path | None): The path to write the prompt to. Default is None.
        - min_examples_required (int): The minimum number of examples required. Default is 2.

    Returns
    -------
    - str: The entity extraction prompt
    """
    if isinstance(entity_types, list):
        entity_types = ", ".join(map(str, entity_types))

    prompt = _select_template(entity_types, json_mode)

    tokenizer = tokenizer or get_tokenizer()

    base_token_count = count_base_prompt_tokens(
        entity_types, language, tokenizer, json_mode
    )

    examples_prompt, examples_used, required_token_count = (
        select_examples_within_budget(
            docs=docs,
            examples=examples,
            entity_types=entity_types,
            base_token_count=base_token_count,
            max_token_count=max_token_count,
            min_examples_required=min_examples_required,
            tokenizer=tokenizer,
        )
    )

    prompt = (
        prompt.format(
            entity_types=entity_types, examples=examples_prompt, language=language
        )
        if entity_types
        else prompt.format(examples=examples_prompt, language=language)
    )

    prompt_token_count = tokenizer.num_tokens(prompt)
    logger.info(
        "Extract graph prompt: %d of %d examples included, %d tokens.",
        examples_used,
        len(examples),
        prompt_token_count,
    )
    if prompt_token_count > max_token_count:
        logger.warning(
            "Extract graph prompt: the %d required examples do not fit in a budget of %d"
            " tokens, the prompt is %d. Use --max-tokens %d to fit all %d generated"
            " examples.",
            examples_used,
            max_token_count,
            prompt_token_count,
            required_token_count,
            len(examples),
        )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        output_path = output_path / EXTRACT_GRAPH_FILENAME
        # Write file to output path
        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))

    return prompt
