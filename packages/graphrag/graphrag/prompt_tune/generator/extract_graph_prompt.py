# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity Extraction prompt generator module."""

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

    tokens_left = max_token_count - count_base_prompt_tokens(
        entity_types, language, tokenizer, json_mode
    )

    examples_prompt = ""

    # Iterate over examples, while we have tokens left or examples left
    for i, output in enumerate(examples):
        input = docs[i]
        example_formatted = (
            EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input, entity_types=entity_types, output=output
            )
            if entity_types
            else UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input, output=output
            )
        )

        example_tokens = tokenizer.num_tokens(example_formatted)

        # Ensure at least three examples are included
        if i >= min_examples_required and example_tokens > tokens_left:
            break

        examples_prompt += example_formatted
        tokens_left -= example_tokens

    prompt = (
        prompt.format(
            entity_types=entity_types, examples=examples_prompt, language=language
        )
        if entity_types
        else prompt.format(examples=examples_prompt, language=language)
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        output_path = output_path / EXTRACT_GRAPH_FILENAME
        # Write file to output path
        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))

    return prompt
