# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from graphrag_input import InputConfig, InputType, create_input_reader
from graphrag_storage import StorageConfig, create_storage


# these tests just confirm we can load files with MarkItDown,
# and use html specifically because it requires no additional dependency installation
async def test_markitdown_loader_one_file():
    config = InputConfig(
        type=InputType.MarkItDown,
        file_pattern=".*\\.html$",
    )
    storage = create_storage(
        StorageConfig(
            base_dir="tests/unit/indexing/input/data/one-html",
        )
    )
    reader = create_input_reader(config, storage)
    documents = await reader.read_files()
    assert len(documents) == 1
    # markitdown will extract the title and body from the HTML if present and clean them
    assert documents[0].title == "Test"
    assert documents[0].text == "Hi how are you today?"
    assert documents[0].raw_data is None


# markitdown guesses the charset from the first 4 KB of the stream, so a file whose
# opening is pure ASCII is misread from the first non-ASCII byte onwards. These tests
# pin that the configured input encoding reaches markitdown instead of being guessed.
ASCII_PREFIX = "Plain ascii line to pad the detection window. " * 120
NON_ASCII_TAIL = "Кальций карбоникум"


async def _read_one_txt(base_dir, **input_config) -> list:
    config = InputConfig(
        type=InputType.MarkItDown,
        file_pattern=".*\\.txt$",
        **input_config,
    )
    storage = create_storage(StorageConfig(base_dir=str(base_dir)))
    reader = create_input_reader(config, storage)
    return await reader.read_files()


async def test_markitdown_loader_uses_configured_encoding(tmp_path):
    (tmp_path / "input.txt").write_bytes(
        f"{ASCII_PREFIX}\n{NON_ASCII_TAIL}\n".encode("windows-1251")
    )

    documents = await _read_one_txt(tmp_path, encoding="windows-1251")

    assert len(documents) == 1
    assert documents[0].text.endswith(f"{NON_ASCII_TAIL}\n")


async def test_markitdown_loader_reads_utf8_file_with_an_ascii_opening(tmp_path):
    # The charset guess lands on ascii, the converter raises on the first cyrillic byte
    # and InputReader swallows it, so the file silently never reaches the index.
    (tmp_path / "input.txt").write_bytes(f"{ASCII_PREFIX}\n{NON_ASCII_TAIL}\n".encode())

    documents = await _read_one_txt(tmp_path)

    assert len(documents) == 1
    assert documents[0].text.endswith(f"{NON_ASCII_TAIL}\n")
