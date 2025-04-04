import pytest

from intrigue import URLAnalyzer


@pytest.fixture(scope="session")
def analyzer():
    """Create a URLAnalyzer instance that's reused across tests."""
    return URLAnalyzer(quiet=True)


@pytest.fixture
def trained_analyzer(analyzer):
    """Create a trained URLAnalyzer instance with sample data."""
    sample_urls = [
        "https://example.com/admin",
        "https://example.com/api/users",
        "https://example.com/static/css/main.css",
        "https://example.com/blog/article",
    ]
    sample_labels = [1, 1, 0, 0]
    analyzer.train(sample_urls, sample_labels)
    return analyzer
