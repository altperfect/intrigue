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


@pytest.fixture
def thorough_analyzer(trained_analyzer):
    """Temporarily disable quick filtering on the analyzer for thorough testing."""
    original_setting = trained_analyzer.use_quick_filter
    trained_analyzer.use_quick_filter = False

    yield trained_analyzer

    trained_analyzer.use_quick_filter = original_setting
