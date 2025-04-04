import pytest

from intrigue import URLAnalyzer


@pytest.fixture
def analyzer():
    """Create a new URLAnalyzer instance for each test."""
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


@pytest.mark.parametrize(
    "url,expected_features",
    [
        (
            "https://example.com/admin/users?role=admin",
            ["ADMIN_AREA"],
        ),
        (
            "https://example.com/api/v1/users/123/permissions",
            ["API_ENDPOINT"],
        ),
        (
            "https://example.com/execute?cmd=ls+-la",
            ["DANGEROUS_PARAM"],
        ),
        (
            "https://example.com/redirect?url=https://evil.com",
            ["SSRF_REDIRECT_URL_IN_PARAM", "SSRF_REDIRECT_CANDIDATE"],
        ),
        (
            "https://example.com/wp-config.php",
            ["SENSITIVE_FILENAME", "WP_CONFIG_FILE"],
        ),
        (
            "https://example.com/assets/main.css",
            ["STATIC_RESOURCE"],
        ),
    ],
)
def test_extract_security_features(analyzer, url, expected_features):
    """Test the feature extraction functionality for various security-relevant URLs."""
    features = analyzer._extract_security_features(url)
    for feature in expected_features:
        assert feature in features


def test_rank_urls_basic_functionality(trained_analyzer):
    """Test that URL ranking returns expected number of results."""
    urls = [
        "https://example.com/",
        "https://example.com/admin/users",
        "https://example.com/assets/main.css",
    ]
    ranked = trained_analyzer.rank_urls(urls)
    assert len(ranked) == len(urls)
    assert all(isinstance(score, float) for _, score in ranked)


@pytest.mark.parametrize(
    "urls,high_interest,low_interest",
    [
        (
            [
                "https://example.com/login?redirect=https://evil.com",
                "https://example.com/backup.sql",
                "https://example.com/admin/users",
                "https://example.com/assets/main.css",
                "https://example.com/",
            ],
            [
                "https://example.com/login?redirect=https://evil.com",
                "https://example.com/backup.sql",
                "https://example.com/admin/users",
            ],
            ["https://example.com/assets/main.css", "https://example.com/"],
        ),
    ],
)
def test_rank_urls_interest_levels(trained_analyzer, urls, high_interest, low_interest):
    """Test that high interest URLs are ranked higher than low interest URLs."""
    ranked = trained_analyzer.rank_urls(urls)
    scores = {url: score for url, score in ranked}

    assert any(url in scores for url in high_interest)

    high_scores = [scores[url] for url in high_interest if url in scores]
    low_scores = [scores[url] for url in low_interest if url in scores]

    if high_scores and low_scores:
        assert max(low_scores) < max(high_scores)
