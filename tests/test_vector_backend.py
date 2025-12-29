import pytest

from src.vector_backend import parse_vector_backend


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("chroma", "chroma"),
        ("Chroma", "chroma"),
        ("chromadb", "chroma"),
        ("cassandra", "cassandra"),
        ("Astra", "cassandra"),
        ("astradb", "cassandra"),
    ],
)
def test_parse_vector_backend(raw: str, expected: str) -> None:
    assert parse_vector_backend(raw) == expected


def test_parse_vector_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_vector_backend("sqlite")
