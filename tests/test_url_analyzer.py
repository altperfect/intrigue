import pytest


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


def test_rank_urls_basic_functionality(thorough_analyzer):
    """Test that URL ranking returns expected number of results."""
    urls = [
        "https://example.com/",
        "https://example.com/admin/users",
        "https://example.com/assets/main.css",
    ]

    ranked = thorough_analyzer.rank_urls(urls)
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
def test_rank_urls_interest_levels(
    thorough_analyzer, urls, high_interest, low_interest
):
    """Test that high interest URLs are ranked higher than low interest URLs."""
    ranked = thorough_analyzer.rank_urls(urls)
    scores = {url: score for url, score in ranked}

    assert any(url in scores for url in high_interest)

    high_scores = [scores[url] for url in high_interest if url in scores]
    low_scores = [scores[url] for url in low_interest if url in scores]

    if high_scores and low_scores:
        assert max(low_scores) < max(high_scores)


@pytest.mark.parametrize(
    "urls,score_threshold",
    [
        (
            [
                "https://example.com/assets/main.css",
                "https://example.com/images/logo.png",
                "https://example.com/styles/theme.css",
            ],
            0.05,
        ),
    ],
)
def test_thorough_mode_static_resource_scoring(
    thorough_analyzer, urls, score_threshold
):
    """Test that static resources are assigned minimal scores with quick filter disabled."""
    ranked = thorough_analyzer.rank_urls(urls, top_n=len(urls))
    scores = {url: score for url, score in ranked}

    assert len(scores) == len(urls)
    for url in urls:
        assert (
            scores[url] < score_threshold
        ), f"Static resource {url} should have a minimal score"


@pytest.mark.parametrize(
    "urls,score_threshold",
    [
        (
            [
                "https://example.com/login?redirect=https://evil.com",
                "https://example.com/admin/users",
            ],
            0.1,
        ),
    ],
)
def test_quick_filter_interesting_resource_scoring(
    trained_analyzer, urls, score_threshold
):
    """Test that interesting resources are assigned high scores with quick filter enabled."""
    ranked = trained_analyzer.rank_urls(urls, top_n=len(urls))
    scores = {url: score for url, score in ranked}

    assert len(scores) == len(urls)
    for url in urls:
        assert (
            scores[url] > score_threshold
        ), f"Interesting resource {url} should have a high score"


def test_quick_filter_filters_static_resources(trained_analyzer):
    """Test that quick filter completely filters out static resources but keeps interesting URLs."""
    static_urls = [
        "https://example.com/assets/main.css",
        "https://example.com/images/logo.png",
        "https://example.com/styles/theme.css",
    ]
    interesting_urls = [
        "https://example.com/login?redirect=https://evil.com",
        "https://example.com/admin/users",
    ]
    all_urls = static_urls + interesting_urls

    trained_analyzer.use_quick_filter = True

    ranked = trained_analyzer.rank_urls(all_urls)
    result_urls = [url for url, _ in ranked]

    for url in static_urls:
        assert url not in result_urls, f"Static resource {url} should be filtered out"

    for url in interesting_urls:
        assert url in result_urls, f"Interesting URL {url} should be kept"
