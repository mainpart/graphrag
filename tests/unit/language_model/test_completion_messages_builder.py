# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for CompletionMessagesBuilder."""

from graphrag_llm.utils import CompletionMessagesBuilder


def test_build_returns_a_copy():
    builder = CompletionMessagesBuilder().add_system_message("persona")

    first = builder.add_user_message("doc 1").build()
    second = builder.add_user_message("doc 2").build()

    assert first is not second
    assert len(first) == 2
    assert len(second) == 3
