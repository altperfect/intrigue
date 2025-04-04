from urllib.parse import parse_qs, urlparse

import pytest


def test_extract_path_context(analyzer):
    """Test path context extraction functions."""
    static_path = "/static/images/logo.png"
    static_ctx = analyzer._extract_path_context(static_path)
    assert static_ctx["is_static_override"] is True

    admin_path = "/admin/users"
    admin_ctx = analyzer._extract_path_context(admin_path)
    assert admin_ctx["is_wp_admin_path"] is False

    wp_admin_path = "/wp-admin/users.php"
    wp_ctx = analyzer._extract_path_context(wp_admin_path)
    assert wp_ctx["is_wp_admin_path"] is True

    support_path = "/support/faq"
    support_ctx = analyzer._extract_path_context(support_path)
    assert support_ctx["is_support_path"] == "SUPPORT_PATH"


def test_extract_tracking_features(analyzer):
    """Test tracking feature extraction."""
    utm_url = "https://example.com/landing?utm_source=google&utm_medium=cpc"
    parsed = urlparse(utm_url)
    params = parse_qs(parsed.query)
    tracking = analyzer._extract_tracking_features(parsed.path, params)
    assert "UTM_PARAMS" in tracking

    analytics_url = "https://example.com/analytics/track?id=123"
    parsed = urlparse(analytics_url)
    params = parse_qs(parsed.query)
    tracking = analyzer._extract_tracking_features(parsed.path, params)
    assert "TRACKING_ENDPOINT" in tracking


def test_extract_high_priority_features(analyzer):
    """Test high priority feature extraction."""
    ssrf_url = "https://example.com/redirect?url=https://evil.com"
    parsed = urlparse(ssrf_url)
    params = parse_qs(parsed.query)
    high_prio = analyzer._extract_high_priority_features(parsed.path, params, [])
    assert "SSRF_REDIRECT_URL_IN_PARAM" in high_prio
    assert "SSRF_REDIRECT_CANDIDATE" in high_prio

    cmd_url = "https://example.com/execute?cmd=ls+-la"
    parsed = urlparse(cmd_url)
    params = parse_qs(parsed.query)
    high_prio = analyzer._extract_high_priority_features(parsed.path, params, [])
    assert any("DANGEROUS_PARAM" in feature for feature in high_prio)


def test_extract_file_features(analyzer):
    """Test file feature extraction."""
    config_url = "https://example.com/wp-config.php"
    parsed = urlparse(config_url)
    path_context = analyzer._extract_path_context(parsed.path)
    file_features = analyzer._extract_file_features(
        parsed.path, "wp-config.php", ".php", [], path_context
    )
    assert any("SENSITIVE_FILENAME" in feature for feature in file_features)
    assert "WP_CONFIG_FILE" in file_features

    backup_url = "https://example.com/backup.sql"
    parsed = urlparse(backup_url)
    path_context = analyzer._extract_path_context(parsed.path)
    file_features = analyzer._extract_file_features(
        parsed.path, "backup.sql", ".sql", [], path_context
    )
    assert any("HIGH_PRIORITY_EXT" in feature for feature in file_features)


def test_extract_auth_features(analyzer):
    """Test authentication feature extraction."""
    oauth_url = "https://example.com/oauth/authorize?client_id=123&response_type=code"
    parsed = urlparse(oauth_url)
    params = parse_qs(parsed.query)
    auth_features = analyzer._extract_auth_features(parsed.path, params, [], [])
    assert "OAUTH_SAML_FLOW" in auth_features


@pytest.mark.parametrize(
    "url,expected_features",
    [
        (
            "https://example.com/api/v1/users",
            ["API_ENDPOINT"],
        ),
        (
            "https://example.com/api/v1/tokens",
            ["API_ENDPOINT", "API_SENSITIVE_KEYWORD"],
        ),
    ],
)
def test_extract_api_features(analyzer, url, expected_features):
    """Test API feature extraction."""
    parsed = urlparse(url)
    path_context = analyzer._extract_path_context(parsed.path)
    api_features = analyzer._extract_api_features(parsed.path, [], path_context)

    for feature in expected_features:
        assert feature in api_features


def test_extract_static_features(analyzer):
    """Test static feature extraction."""
    static_url = "https://example.com/assets/styles.css"
    parsed = urlparse(static_url)
    path_context = analyzer._extract_path_context(parsed.path)
    static_features = analyzer._extract_static_features(
        parsed.path, ".css", "styles.css", [], path_context, []
    )
    assert "STATIC_RESOURCE" in static_features
    assert "STATIC_RESOURCE" in static_features
