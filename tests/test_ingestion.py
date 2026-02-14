"""Basic tests for document ingestion."""
from src.ingestion.preprocessor import clean_text


def test_clean_text():
    text = "  Hello   world  "
    assert clean_text(text) == "Hello world"
