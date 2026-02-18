"""Load documents from files."""

def load_text_file(file_path: str) -> str:
    """Load a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """Load a PDF file."""
    # TODO: Implement PDF loading
    pass
