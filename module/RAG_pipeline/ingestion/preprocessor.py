"""Basic text preprocessing."""

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.strip()
    text = ' '.join(text.split())
    return text
