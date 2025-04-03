import argparse
import io
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from constants import BASE_DOMAINS, INTERESTING_PARAMS, INTERESTING_PATHS


class URLAnalyzer:
    """
    Analyzer for ranking URLs by security interest.

    Uses a TF-IDF vectorizer with character n-grams followed by a Random Forest
    classifier to identify potentially interesting URLs for security testing.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the URL analyzer with an optional pre-trained model.

        Args:
            model_path: Path to a saved model file. If None, a new model will be created.
        """
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self._create_new_model()
        else:
            self._create_new_model()

        self.MAX_REPETITIONS = {
            "OAUTH_SAML_FLOW": 30,
            "HASHED_JS_FILE": 2,
            "CERTIFICATE_PRIVATE_PATH": 30,
            "API_ENDPOINT": 30,
            "DEFAULT": 1,
        }

    def _create_new_model(self) -> None:
        """Create a new pipeline for URL analysis."""
        self.model = Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 6),
                        lowercase=True,
                        max_features=10000,
                        use_idf=True,
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=25,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    def train(self, urls: List[str], labels: List[int]) -> None:
        """
        Train the model with URLs and their corresponding labels.

        Args:
            urls: List of URLs to train on
            labels: Binary labels (1 for interesting, 0 for not interesting)
        """
        print(f"Training model on {len(urls)} URLs...")

        enhanced_urls = []
        for url in urls:
            features = self._extract_security_features(url)
            enhanced_url = f"{url} {features}"
            enhanced_urls.append(enhanced_url)

        self.model.fit(enhanced_urls, labels)
        print("Training complete")

    def _extract_security_features(self, url: str) -> str:
        """Extract security-specific features from a URL and format as a string."""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            query = parsed.query
            params = parse_qs(query)
        except ValueError:
            return "INVALID_URL"

        features = []
        file_ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path).lower()

        # static/content path identification (early check)
        static_indicators = [
            "/icons/",
            "/images/",
            "/img/",
            "/static/",
            "/assets/",
            "/css/",
            "/fonts/",
            "/svg/",
            "/common-v2/",
        ]
        content_paths = [
            "/blog/",
            "/news/",
            "/articles/",
            "/post/",
            "/category/",
            "/tag/",
            "/event/",
        ]
        support_paths = ["/support/", "/help/", "/faq/"]
        dev_docs_paths = [
            "/docs/",
            "/sdk/",
            "/api-docs/",
            "/developer/",
            "/documentation/",
            "/reference/",
        ]
        survey_paths = ["/survey", "/feedback"]
        is_likely_static_path = any(
            indicator in path for indicator in static_indicators
        )
        is_svg_icon_path = any(indicator in path for indicator in ["/icons/", "/svg/"])
        is_content_path = any(cp in path for cp in content_paths)
        is_support_path = any(sp in path for sp in support_paths)
        is_dev_docs_path = any(dp in path for dp in dev_docs_paths)
        is_survey_path = any(sp in path for sp in survey_paths)
        is_static_override = is_likely_static_path or is_svg_icon_path

        # tracking/metrics/utm detection
        tracking_paths = [
            "/track-event",
            "/pixel",
            "/beacon",
            "/analytics",
            "/metrics",
            "/collect",
            "/track/",
        ]
        has_utm_params = any(p.startswith("utm_") for p in params.keys())
        is_tracking = any(tp in path for tp in tracking_paths)

        if is_tracking:
            features.append("TRACKING_ENDPOINT")
        if has_utm_params:
            features.append("UTM_PARAMS")
            is_tracking = True

        if is_static_override:
            features.append("IS_STATIC_OVERRIDE")

        # high priority features
        redirect_params = [
            "redirect",
            "redirect_uri",
            "return",
            "returnurl",
            "back",
            "next",
            "url",
            "target",
            "dest",
            "goto",
            "continue",
        ]
        is_ssrf_or_redirect_candidate = False
        for param_name in params.keys():
            if any(
                redirect_term in param_name.lower() for redirect_term in redirect_params
            ):
                for value in params[param_name]:
                    if (
                        "://" in value
                        or value.startswith("http")
                        or value.startswith("//")
                        or "%3a%2f%2f" in value.lower()
                    ):
                        features.append("SSRF_REDIRECT_URL_IN_PARAM")
                        is_ssrf_or_redirect_candidate = True
                        break
                if (
                    is_ssrf_or_redirect_candidate
                    and "TRACKING_ENDPOINT" not in features
                ):
                    features.append("SSRF_REDIRECT_CANDIDATE")
                    break

        dangerous_params = [
            "cmd",
            "exec",
            "command",
            "run",
            "system",
            "shell",
            "ping",
            "sql",
            "select",
            "insert",
            "update",
            "delete",
            "union",
        ]
        simple_query_params = ["query", "search", "q", "keyword"]
        for param_name in params.keys():
            param_lower = param_name.lower()
            if param_lower in dangerous_params:
                if not is_tracking:
                    features.append(f"DANGEROUS_PARAM:{param_lower}")
                    for value in params[param_name]:
                        if any(
                            char in value
                            for char in ["'", ";", "|", "`", "$", "(", ")"]
                        ):
                            features.append("SUSPICIOUS_PARAM_VALUE")
                            break
            elif param_lower in simple_query_params:
                if not is_tracking:
                    features.append(f"SIMPLE_QUERY_PARAM:{param_lower}")

        high_priority_exts = [
            ".sql",
            ".bak",
            ".backup",
            ".env",
            ".config",
            ".conf",
            ".yml",
            ".yaml",
            ".pem",
            ".key",
            ".p12",
        ]
        medium_priority_exts = [
            ".zip",
            ".tar",
            ".gz",
            ".log",
            ".txt",
            ".csv",
            ".json",
            ".xml",
        ]
        low_priority_exts = [".pdf", ".doc", ".docx", ".xls", ".xlsx"]
        sensitive_filenames = [
            ".git",
            ".svn",
            ".htaccess",
            "web.config",
            "docker-compose",
            "makefile",
            "id_rsa",
            "credentials",
            "wp-config.php",
        ]
        wp_admin_sensitive_files = [
            "options.php",
            "settings.php",
            "admin.php",
            "user-edit.php",
            "plugin-install.php",
            "theme-editor.php",
        ]

        is_wp_admin_path = path.startswith("/wp-admin/")

        if file_ext in high_priority_exts:
            if "TRACKING_ENDPOINT" not in features:
                features.append(f"HIGH_PRIORITY_EXT:{file_ext}")
        elif file_ext in medium_priority_exts:
            if file_ext in [".json", ".xml"] and any(
                sp in path for sp in ["/static/", "/assets/"]
            ):
                features.append("STATIC_DATA_FILE")
            else:
                if "TRACKING_ENDPOINT" not in features:
                    features.append(f"MEDIUM_PRIORITY_EXT:{file_ext}")
        elif file_ext in low_priority_exts:
            if "TRACKING_ENDPOINT" not in features:
                features.append(f"LOW_PRIORITY_EXT:{file_ext}")

        if filename in sensitive_filenames:
            if "TRACKING_ENDPOINT" not in features:
                features.append(f"SENSITIVE_FILENAME:{filename}")
            if filename == "wp-config.php":
                features.append("WP_CONFIG_FILE")

        file_paths = [
            p
            for p in INTERESTING_PATHS
            if any(
                x in p
                for x in [
                    "/download",
                    "/file",
                    "/export",
                    "/backup",
                    "/include",
                    "/static",
                ]
            )
        ]
        if any(fp in path for fp in file_paths):
            features.append("FILE_ACCESS_PATH")
            if (
                "../" in url
                or "%2e%2e%2f" in url.lower()
                or "..\\" in url
                or "%2e%2e\\" in url.lower()
            ):
                features.append("PATH_TRAVERSAL_PATTERN")

        if is_wp_admin_path and filename in wp_admin_sensitive_files:
            if "TRACKING_ENDPOINT" not in features:
                features.append("WP_ADMIN_SENSITIVE_FILE")

        # medium priority features
        oidc_oauth_paths = [
            "/connect/authorize",
            "/oauth/authorize",
            "/oidc",
            "/saml",
            "/login",
            "/signin",
            "/auth",
        ]
        oidc_oauth_params = [
            "client_id",
            "response_type",
            "scope",
            "state",
            "nonce",
            "samlrequest",
            "samlresponse",
        ]
        is_oidc_flow = False
        if (
            "TRACKING_ENDPOINT" not in features
            and any(p in path for p in oidc_oauth_paths)
            or any(p in params for p in oidc_oauth_params)
        ):
            features.append("OAUTH_SAML_FLOW")
            is_oidc_flow = True

        if "TRACKING_ENDPOINT" not in features and path.startswith("/.well-known/"):
            if filename == "openid-configuration":
                if not is_oidc_flow:
                    features.append("WELL_KNOWN_OPENID_CONFIG")
            elif filename == "security.txt":
                features.append("WELL_KNOWN_SECURITY_TXT")
            else:
                features.append("WELL_KNOWN_STATIC")

        if not is_tracking and file_ext == ".js":
            if re.search(r"\.[a-f0-9]{8,}\.js$", path):
                features.append("HASHED_JS_FILE")
            elif is_dev_docs_path:
                features.append("DOCS_STATIC_JS")
            elif "HASHED_JS_FILE" not in features:
                common_js = [
                    "jquery",
                    "bootstrap",
                    "react",
                    "angular",
                    "vue",
                    "app",
                    "main",
                    "script",
                    "bundle",
                    "chunk",
                    "vendor",
                ]
                is_common_name = any(common in filename for common in common_js)
                is_common_path = any(
                    p in path for p in ["/js/", "/static/", "/assets/", "/dist/"]
                )
                if not (is_common_name and is_common_path):
                    features.append("UNUSUAL_JS_FILE")
                else:
                    features.append("COMMON_JS_FILE")
            else:
                features.append("COMMON_JS_FILE")

        admin_paths = [
            p
            for p in INTERESTING_PATHS
            if any(
                x in p
                for x in [
                    "/admin",
                    "/manage",
                    "/console",
                    "/panel",
                    "/dashboard",
                    "/internal",
                    "/private",
                    "/secure",
                ]
            )
        ]
        is_wp_admin_path = path.startswith("/wp-admin/")
        if any(ap in path for ap in admin_paths) and is_likely_static_path:
            features.append("STATIC_ADMIN_INDICATOR")
        elif is_wp_admin_path:
            if not is_tracking:
                features.append("WP_ADMIN_AREA")
        elif any(ap in path for ap in admin_paths):
            if not is_tracking:
                features.append("ADMIN_AREA")

        if "TRACKING_ENDPOINT" not in features and (
            "/api/" in path or "/rest/" in path or "/graphql" in path
        ):
            features.append("API_ENDPOINT")

        if is_dev_docs_path and not is_tracking:
            if any(kw in path for kw in ["/com/", "/objects/", "/tools/"]):
                features.append("DEV_DOCS_TOOLS_PATH")
            else:
                features.append("DEV_DOCS_GENERIC_PATH")

        # low priority / context / penalties

        if is_support_path:
            features.append("SUPPORT_PATH")

        if is_survey_path and "TRACKING_ENDPOINT" not in features:
            features.append("SURVEY_PATH")

        if path.endswith("/certificate/private"):
            features.append("CERTIFICATE_PRIVATE_PATH")

        if path.endswith("/managers"):
            features.append("MANAGERS_PATH")

        if filename == "robots.txt":
            features.append("ROBOTS_TXT")

        common_doc_names = [
            "license",
            "privacy",
            "terms",
            "oferta",
            "offer",
            "security-policy",
        ]
        if filename.split(".")[0] in common_doc_names:
            features.append("COMMON_DOC_FILENAME")

        # final static/content page identification
        static_exts = [
            ".css",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".ico",
            ".html",
            ".htm",
        ]
        if (
            not is_tracking
            and "COMMON_JS_FILE" not in features
            and "HASHED_JS_FILE" not in features
            and "ROBOTS_TXT" not in features
            and "COMMON_DOC_FILENAME" not in features
        ):
            if "IS_STATIC_OVERRIDE" in features and file_ext == ".svg":
                features.append("ICON_SVG_STATIC")
            elif is_static_override or file_ext in static_exts:
                if file_ext in [".html", ".htm"]:
                    if is_dev_docs_path:
                        features.append("HTML_DOC_IN_DEV_PATH")
                    else:
                        features.append("PLAIN_HTML_FILE")
                elif "ICON_SVG_STATIC" not in features:
                    features.append("STATIC_RESOURCE")

            elif is_content_path:
                features.append("CONTENT_PAGE")

        return " ".join(features)

    def rank_urls(self, urls: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Rank URLs by their security interestingness.

        Args:
            urls: List of URLs to analyze
            top_n: Number of top results to return

        Returns:
            List of (url, score) tuples sorted by score in descending order
        """
        BATCH_SIZE = 5000
        URL_THRESHOLD = 50000

        if not urls:
            return []

        try:
            unique_urls = list(dict.fromkeys(urls))

            if len(unique_urls) > URL_THRESHOLD:
                msg = f"Processing large URL set ({len(unique_urls)} URLs). This may take some time..."
                print(msg)

            all_scores = []

            for i in range(0, len(unique_urls), BATCH_SIZE):
                batch = unique_urls[i : i + BATCH_SIZE]  # noqa: E203

                batch_enhanced = []
                for url in batch:
                    features = self._extract_security_features(url)
                    enhanced_url = f"{url} {features}"
                    batch_enhanced.append(enhanced_url)

                batch_probs = self.model.predict_proba(batch_enhanced)

                batch_scores = [(url, prob[1]) for url, prob in zip(batch, batch_probs)]
                all_scores.extend(batch_scores)

            ranked_diverse = self._ensure_diversity(all_scores)

            return ranked_diverse[:top_n]
        except NotFittedError:
            raise NotFittedError(
                "The model has not been trained yet. Please train the model first."
            )

    def _sample_diverse_urls(self, urls: List[str], max_count: int) -> List[str]:
        """
        Sample a diverse subset of URLs when dealing with very large datasets.
        Ensures representation across different domains and path types.

        Args:
            urls: Full list of URLs to sample from
            max_count: Maximum number of URLs to return

        Returns:
            A diverse sample of URLs
        """
        domains = {}
        for url in urls:
            domain = urlparse(url).netloc
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(url)

        domains_count = len(domains)
        if domains_count == 0:
            return []

        urls_per_domain = max(1, max_count // domains_count)

        result = []
        for domain, domain_urls in domains.items():
            sample_size = min(len(domain_urls), urls_per_domain)

            interesting_paths = [
                u
                for u in domain_urls
                if any(
                    p in u.lower()
                    for p in [
                        "/admin",
                        "/api",
                        "/auth",
                        "/login",
                        "/.well-known",
                        "/oauth",
                        "/openid",
                        "/config",
                        "/settings",
                        "/upload",
                        "/download",
                    ]
                )
            ]

            if len(interesting_paths) > sample_size:
                domain_sample = random.sample(interesting_paths, sample_size)
            else:
                domain_sample = interesting_paths
                remaining_urls = [u for u in domain_urls if u not in interesting_paths]
                remaining_needed = sample_size - len(domain_sample)
                if remaining_needed > 0 and remaining_urls:
                    domain_sample.extend(
                        random.sample(
                            remaining_urls, min(len(remaining_urls), remaining_needed)
                        )
                    )

            result.extend(domain_sample)

        if len(result) < max_count:
            remaining_needed = max_count - len(result)
            large_domains = sorted(
                domains.items(), key=lambda x: len(x[1]), reverse=True
            )

            for domain, domain_urls in large_domains:
                if remaining_needed <= 0:
                    break

                unused_urls = [u for u in domain_urls if u not in result]
                to_add = min(remaining_needed, len(unused_urls))

                if to_add > 0:
                    result.extend(random.sample(unused_urls, to_add))
                    remaining_needed -= to_add

        return result

    def _ensure_diversity(
        self, url_scores: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Ensure diversity in results by prioritizing URLs based on specific security patterns."""

        feature_weights = {
            # high priority
            "SSRF_REDIRECT_CANDIDATE": 0.32,
            "DANGEROUS_PARAM": 0.30,
            "PATH_TRAVERSAL_PATTERN": 0.25,
            "SENSITIVE_FILENAME": 0.22,
            "HIGH_PRIORITY_EXT": 0.18,
            "SUSPICIOUS_PARAM_VALUE": 0.10,
            "WP_CONFIG_FILE": 0.30,
            "WP_ADMIN_SENSITIVE_FILE": 0.22,
            # medium priority
            "ADMIN_AREA": 0.14,
            "WP_ADMIN_AREA": 0.13,
            "OAUTH_SAML_FLOW": 0.14,
            "MEDIUM_PRIORITY_EXT": 0.06,
            "UNUSUAL_JS_FILE": 0.07,
            "API_ENDPOINT": 0.05,
            "DEV_DOCS_TOOLS_PATH": 0.16,
            "DEV_DOCS_GENERIC_PATH": 0.05,
            "WELL_KNOWN_OPENID_CONFIG": 0.03,
            "WELL_KNOWN_SECURITY_TXT": -0.75,
            # low priority / context
            "FILE_ACCESS_PATH": 0.04,
            "LOW_PRIORITY_EXT": 0.01,
            "HAS_PARAMS": 0.01,
            "PARAM_COUNT": 0.005,
            "SIMPLE_QUERY_PARAM": 0.02,
            # Penalties (slightly moderated extreme penalties)
            "TRACKING_ENDPOINT": -0.80,
            "UTM_PARAMS": -0.70,
            "SURVEY_PATH": -0.70,
            "BORING_PARAMS": -0.10,
            "STATIC_RESOURCE": -0.55,
            "PLAIN_HTML_FILE": -0.60,
            "ROBOTS_TXT": -0.80,
            "COMMON_DOC_FILENAME": -0.80,
            "CONTENT_PAGE": -0.45,
            "COMMON_JS_FILE": -0.30,
            "HASHED_JS_FILE": -0.40,
            "SUPPORT_PATH": -0.30,
            "IS_STATIC_OVERRIDE": -0.70,
            "DOCS_STATIC_JS": -0.35,
            "CERTIFICATE_PRIVATE_PATH": -0.30,
            "MANAGERS_PATH": -0.45,
            "ICON_SVG_STATIC": -0.70,
            "HTML_DOC_IN_DEV_PATH": -0.25,
            "STATIC_DATA_FILE": -0.15,
            "WELL_KNOWN_STATIC": -0.35,
            "INVALID_URL": -1.0,
        }

        adjusted_scores = []
        for url, initial_score in url_scores:
            features_str = self._extract_security_features(url)
            features = features_str.split()
            parsed = urlparse(url)

            base_score = 0.15 + (initial_score * 0.35)

            positive_feature_score_sum = 0
            negative_feature_score_sum = 0
            primary_penalty_feature = None
            found_high_priority_feature = False
            has_boring_params = "BORING_PARAMS" in features
            is_oauth = "OAUTH_SAML_FLOW" in features
            is_wp_sensitive = (
                "WP_ADMIN_SENSITIVE_FILE" in features or "WP_CONFIG_FILE" in features
            )
            is_general_admin = "ADMIN_AREA" in features
            is_wp_admin = "WP_ADMIN_AREA" in features
            is_redirect = "SSRF_REDIRECT_CANDIDATE" in features
            is_search = "SIMPLE_QUERY_PARAM" in features
            is_api = "API_ENDPOINT" in features
            is_file_access = "FILE_ACCESS_PATH" in features
            is_unusual_js = "UNUSUAL_JS_FILE" in features

            for feature in features:
                base_feature = feature.split(":")[0]
                weight = feature_weights.get(base_feature)
                if weight is not None:
                    current_score_component = 0
                    if base_feature == "PARAM_COUNT":
                        try:
                            count = int(feature.split(":")[1])
                            current_score_component = weight * min(count, 5)
                        except (IndexError, ValueError):
                            pass
                    else:
                        current_score_component = weight
                    if weight > 0:
                        positive_feature_score_sum += current_score_component
                    else:
                        negative_feature_score_sum += current_score_component
                        if primary_penalty_feature is None and base_feature in [
                            "TRACKING_ENDPOINT",
                            "UTM_PARAMS",
                            "SURVEY_PATH",
                            "ICON_SVG_STATIC",
                            "IS_STATIC_OVERRIDE",
                            "PLAIN_HTML_FILE",
                            "ROBOTS_TXT",
                            "COMMON_DOC_FILENAME",
                            "WELL_KNOWN_SECURITY_TXT",
                        ]:
                            primary_penalty_feature = base_feature
                    if weight >= 0.18:
                        found_high_priority_feature = True

            # scoring logic
            if primary_penalty_feature:
                penalty = feature_weights.get(primary_penalty_feature, -0.7)
                final_score = (
                    base_score * 0.3 + positive_feature_score_sum * 0.2 + penalty
                )
            elif is_oauth:
                final_score = (
                    0.26
                    + base_score * 0.25
                    + positive_feature_score_sum * 0.60
                    + negative_feature_score_sum
                )
            elif is_wp_sensitive:
                final_score = min(
                    0.75,
                    0.35
                    + base_score * 0.15
                    + positive_feature_score_sum * 0.70
                    + negative_feature_score_sum,
                )
            elif is_redirect and is_file_access:
                final_score = min(
                    0.72,
                    0.35
                    + base_score * 0.2
                    + positive_feature_score_sum * 0.85
                    + negative_feature_score_sum,
                )
            elif is_redirect:
                final_score = min(
                    0.68,
                    0.30
                    + base_score * 0.2
                    + positive_feature_score_sum * 0.80
                    + negative_feature_score_sum,
                )
            elif is_general_admin or is_wp_admin:
                final_score = min(
                    0.65,
                    0.26
                    + base_score * 0.2
                    + positive_feature_score_sum * 0.58
                    + negative_feature_score_sum,
                )
            elif found_high_priority_feature:
                final_score = (
                    base_score * 0.5
                    + positive_feature_score_sum
                    + negative_feature_score_sum
                )
            elif is_search:
                final_score = min(
                    0.50,
                    0.25
                    + base_score * 0.25
                    + positive_feature_score_sum * 0.55
                    + negative_feature_score_sum,
                )
            elif is_api and is_search:
                final_score = min(
                    0.40,
                    0.20
                    + base_score * 0.25
                    + positive_feature_score_sum * 0.50
                    + negative_feature_score_sum,
                )
            elif is_api:
                final_score = min(
                    0.35,
                    0.15
                    + base_score * 0.25
                    + positive_feature_score_sum * 0.40
                    + negative_feature_score_sum,
                )
            elif is_unusual_js:
                final_score = min(
                    0.45,
                    0.15
                    + base_score * 0.2
                    + positive_feature_score_sum * 0.6
                    + negative_feature_score_sum,
                )
            else:
                final_score = (
                    base_score * 0.90
                    + positive_feature_score_sum * 0.45
                    + negative_feature_score_sum
                )

            if (
                has_boring_params
                and not found_high_priority_feature
                and primary_penalty_feature is None
            ):
                final_score += feature_weights["BORING_PARAMS"]

            final_score += random.uniform(-0.005, 0.005)
            final_score = max(0.01, min(0.95, final_score))
            adjusted_scores.append((url, final_score))

        ranked = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)

        # Diversity Logic (uses updated MAX_REPETITIONS from __init__)
        result = []
        pattern_counts = defaultdict(int)
        feature_counts = defaultdict(int)

        for url, score in ranked:
            if score < 0.04:
                continue

            parsed = urlparse(url)
            domain = parsed.netloc
            path_parts = [p for p in parsed.path.lower().split("/") if p]
            features_str = self._extract_security_features(url)
            features = features_str.split()

            primary_feature_key = "DEFAULT"
            for key in self.MAX_REPETITIONS:
                if key in features:
                    primary_feature_key = key
                    break

            part1 = path_parts[0] if len(path_parts) > 0 else ""
            part2 = path_parts[1] if len(path_parts) > 1 else ""
            part3 = path_parts[2] if len(path_parts) > 2 else ""
            path_pattern_key = f"{domain}:{part1}:{part2}:{part3}"

            max_feature_allowed = self.MAX_REPETITIONS.get(
                primary_feature_key, self.MAX_REPETITIONS["DEFAULT"]
            )

            is_heavily_penalized = any(
                f in features
                for f in [
                    "ROBOTS_TXT",
                    "COMMON_DOC_FILENAME",
                    "WELL_KNOWN_SECURITY_TXT",
                    "PLAIN_HTML_FILE",
                    "ICON_SVG_STATIC",
                    "SURVEY_PATH",
                    "IS_STATIC_OVERRIDE",
                    "TRACKING_ENDPOINT",
                    "UTM_PARAMS",
                ]
            )
            max_path_allowed = 1 if (score < 0.2 or is_heavily_penalized) else 4

            current_feature_count = feature_counts.get(primary_feature_key, 0)
            current_path_count = pattern_counts.get(path_pattern_key, 0)

            if (
                current_feature_count >= max_feature_allowed
                or current_path_count >= max_path_allowed
            ):
                continue

            result.append((url, score))
            feature_counts[primary_feature_key] = current_feature_count + 1
            pattern_counts[path_pattern_key] = current_path_count + 1

        return sorted(result, key=lambda x: x[1], reverse=True)

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            model_path: Path where the model will be saved
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract security-relevant features from a URL.

    Args:
        url: URL to analyze

    Returns:
        Dictionary of features extracted from the URL
    """
    parsed = urlparse(url)
    path = parsed.path
    query = parsed.query
    params = parse_qs(query)

    file_ext = os.path.splitext(path)[1].lower()

    features = {
        "domain": parsed.netloc,
        "path": path,
        "has_query": len(query) > 0,
        "num_params": len(params),
        "file_ext": file_ext,
        "param_names": list(params.keys()),
        "path_depth": len([p for p in path.split("/") if p]),
        "contains_ip": bool(re.search(r"\d+\.\d+\.\d+\.\d+", parsed.netloc)),
        "contains_port": bool(parsed.port),
    }

    for pattern in INTERESTING_PATHS:
        features[f'has_path_{pattern.replace("/", "_")}'] = pattern in path.lower()

    for param in INTERESTING_PARAMS:
        features[f"has_param_{param}"] = param in params

    return features


def process_url_file(file_path: str) -> List[str]:
    """
    Read URLs from a file, one per line.

    Args:
        file_path: Path to the file containing URLs

    Returns:
        List of URLs read from the file
    """
    URL_READ_PROGRESS_INTERVAL = 100000
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB in bytes

    try:
        file_size = os.path.getsize(file_path)

        # Use buffered reading for large files
        if file_size > LARGE_FILE_THRESHOLD:
            urls = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        urls.append(line)

                    if len(urls) % URL_READ_PROGRESS_INTERVAL == 0:
                        print(f"Read {len(urls)} URLs so far...")

            return urls
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []


def generate_sample_data(output_file: str, num_samples: int = 100) -> str:
    """
    Generate sample training data with labeled URLs, emphasizing security patterns.

    Args:
        output_file: Path where the sample data will be saved
        num_samples: Number of sample URLs to generate

    Returns:
        Path to the generated sample file
    """
    interesting_count = num_samples // 3
    normal_count = num_samples - interesting_count

    interesting_urls = []

    pattern_categories = {
        "SSRF_REDIRECT": [
            lambda: f"{random.choice(BASE_DOMAINS)}/redirect?url=https://{random.choice(BASE_DOMAINS)}/profile",
            lambda: f"{random.choice(BASE_DOMAINS)}/login?back=//{random.choice(BASE_DOMAINS)}/admin",
            lambda: f"{random.choice(BASE_DOMAINS)}/auth?next=http%3A%2F%2Fevil.com",
            lambda: f"{random.choice(BASE_DOMAINS)}/goto?dest=https://partner-site.com/login",
            lambda: f"{random.choice(BASE_DOMAINS)}/oauth/authorize?redirect_uri=https://app.com/callback&state=xyz",
        ],
        "DANGEROUS_PARAMS": [
            lambda: f"{random.choice(BASE_DOMAINS)}/search?query=' OR 1=1--",
            lambda: f"{random.choice(BASE_DOMAINS)}/api/users?id=1 UNION SELECT null, username, password FROM users--",
            lambda: f"{random.choice(BASE_DOMAINS)}/tools/ping?ip=127.0.0.1; ls -la",
            lambda: f"{random.choice(BASE_DOMAINS)}/exec?cmd=cat /etc/passwd",
            lambda: f"{random.choice(BASE_DOMAINS)}/debug?command=whoami",
        ],
        "INTERESTING_FILES": [
            lambda: f"{random.choice(BASE_DOMAINS)}/download?file=../../../etc/shadow",
            lambda: f"{random.choice(BASE_DOMAINS)}/backup/db_backup_{random.randint(2020, 2024)}.sql.gz",
            lambda: f"{random.choice(BASE_DOMAINS)}/config/settings.yml",
            lambda: f"{random.choice(BASE_DOMAINS)}/.git/config",
            lambda: f"{random.choice(BASE_DOMAINS)}/files/report_{random.randint(1, 100)}.pdf",
            lambda: f"{random.choice(BASE_DOMAINS)}/export?format=json&data=users",
        ],
        "OAUTH_OIDC": [
            lambda: f"{random.choice(BASE_DOMAINS)}/connect/authorize?client_id=app&response_type=code"
            f"&scope=openid&state=abc",
            lambda: f"{random.choice(BASE_DOMAINS)}/.well-known/openid-configuration",
            lambda: f"{random.choice(BASE_DOMAINS)}/saml/sso?entityID=partner",
        ],
        "ADMIN_UNUSUAL_JS": [
            lambda: f"{random.choice(BASE_DOMAINS)}/admin/dashboard.js",
            lambda: f"{random.choice(BASE_DOMAINS)}/static/admin-utils-{random.randint(100, 999)}.js",
            lambda: f"{random.choice(BASE_DOMAINS)}/management/console",
            lambda: f"{random.choice(BASE_DOMAINS)}/secure/settings/panel",
            lambda: f"{random.choice(BASE_DOMAINS)}/internal/api/debug",
        ],
        "API": [
            lambda: f"{random.choice(BASE_DOMAINS)}/api/v2/users/{random.randint(1, 1000)}/orders",
            lambda: f"{random.choice(BASE_DOMAINS)}/rest/products?category=electronics",
            lambda: f"{random.choice(BASE_DOMAINS)}/graphql?query={{me{{id}}}}",
        ],
    }

    category_weights = {
        "SSRF_REDIRECT": 0.25,
        "DANGEROUS_PARAMS": 0.20,
        "INTERESTING_FILES": 0.20,
        "OAUTH_OIDC": 0.10,
        "ADMIN_UNUSUAL_JS": 0.15,
        "API": 0.10,
    }

    total_weight = sum(category_weights.values())
    normalized_weights = {k: v / total_weight for k, v in category_weights.items()}

    category_list = list(normalized_weights.keys())
    weights_list = list(normalized_weights.values())

    for _ in range(interesting_count):
        chosen_category = random.choices(category_list, weights=weights_list, k=1)[0]

        url_generator = random.choice(pattern_categories[chosen_category])
        url = "https://" + url_generator()
        interesting_urls.append((url, 1))

    # Generate normal URLs
    normal_urls = []
    normal_categories = [
        lambda: random.choice(BASE_DOMAINS)
        + "/css/"
        + random.choice(["main.css", "style.css", "theme.css"]),
        lambda: random.choice(BASE_DOMAINS)
        + "/js/"
        + random.choice(["app.js", "main.js", "vendor.js"]),
        lambda: random.choice(BASE_DOMAINS)
        + "/images/"
        + random.choice(["logo.png", "banner.jpg", "icon.svg"]),
        lambda: random.choice(BASE_DOMAINS)
        + random.choice(["/", "/home", "/about", "/contact", "/products"]),
        lambda: (
            f"{random.choice(BASE_DOMAINS)}/blog/post-{random.randint(1, 50)}"
            + ("?page=" + str(random.randint(1, 5)) if random.random() > 0.8 else "")
        ),
        lambda: random.choice(BASE_DOMAINS)
        + random.choice(["/faq", "/help", "/terms", "/privacy"]),
    ]
    for _ in range(normal_count):
        pattern_gen = random.choice(normal_categories)
        url = "https://" + pattern_gen()
        if random.random() > 0.9:
            url += f"?ref={random.choice(['google', 'newsletter', 'social'])}&utm_campaign=promo"
        normal_urls.append((url, 0))

    all_urls = interesting_urls + normal_urls
    random.shuffle(all_urls)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(all_urls, columns=["url", "is_interesting"])
    df.to_csv(output_file, index=False)

    return output_file


def prompt_yes_no(question: str) -> bool:
    """
    Prompt the user with a yes/no question.

    Args:
        question: The question to ask

    Returns:
        True if the user answered yes, False otherwise
    """
    valid_responses = {"yes": True, "y": True, "no": False, "n": False}

    while True:
        sys.stdout.write(f"{question} [y/n] ")
        choice = input().lower()
        if choice in valid_responses:
            return valid_responses[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def display_training_instructions() -> None:
    """Display instructions for training a new model."""
    print(
        """
Error: No trained model found or model failed to load.

To train a new model, you can:
1. Generate sample training data:
   python intrigue.py --generate-sample --sample-size 1000

2. Train the model with the generated data:
   python intrigue.py --train --train-file data/sample_training_data.csv

For better results, consider creating your own training data with real URLs labeled as interesting (1) or not (0).
"""
    )


def setup_utf8_stdout():
    """Attempt to configure stdout for UTF-8 output with line buffering."""
    try:
        # Use line_buffering=True to flush output on newlines
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except Exception as e:
        print(f"Warning: Could not configure stdout for UTF-8: {e}", file=sys.stderr)


def main() -> None:
    """
    Command-line interface for the URL analyzer.

    Parses command-line arguments and runs the appropriate analyzer functions
    based on the specified options.
    """
    try:
        parser = argparse.ArgumentParser(
            description="ML-based URL analyzer for security testing"
        )
        parser.add_argument("-f", "--file", help="File containing URLs (one per line)")
        parser.add_argument("-u", "--url", help="Single URL to analyze")
        parser.add_argument(
            "-m",
            "--model",
            default="models/url_model.joblib",
            help="Path to model file",
        )
        parser.add_argument(
            "-n", "--top", type=int, default=10, help="Number of top URLs to return"
        )
        parser.add_argument(
            "--train", action="store_true", help="Train a new model with labeled data"
        )
        parser.add_argument(
            "--train-file", help="CSV file with URLs and labels for training"
        )
        parser.add_argument(
            "--generate-sample",
            action="store_true",
            help="Generate sample training data and exit",
        )
        parser.add_argument(
            "--sample-size",
            type=int,
            default=100,
            help="Number of sample URLs to generate (when using --generate-sample)",
        )
        parser.add_argument(
            "--sample-output",
            default="data/sample_training_data.csv",
            help="Output file for sample data (when using --generate-sample)",
        )
        parser.add_argument(
            "--quiet", action="store_true", help="Suppress progress messages"
        )

        args = parser.parse_args()

        if not args.quiet:
            setup_utf8_stdout()

        # Handle sample data generation
        if args.generate_sample:
            sample_file = generate_sample_data(args.sample_output, args.sample_size)
            print(
                f"Generated sample training data with {args.sample_size} URLs at: {sample_file}\n"
                "To train the model with this data, run:\n"
                f"python intrigue.py --train --train-file {sample_file}"
            )
            return

        # Check for stdin input early
        stdin_has_data = not sys.stdin.isatty()

        # Generate sample data if it doesn't exist and we're not training explicitly
        sample_file = "data/sample_training_data.csv"
        if not os.path.exists(sample_file) and not args.train:
            # If we're using stdin for URLs, generate sample data without prompting
            if stdin_has_data:
                # We're using stdin for URL data, just generate the sample without asking
                print("No training data found. Generating sample data for future use.")
                os.makedirs(os.path.dirname(sample_file), exist_ok=True)
                generate_sample_data(sample_file)
                print(f"Generated sample training data at: {sample_file}")
            else:
                # Interactive mode, prompt user
                generate_sample = prompt_yes_no(
                    "No training data found. Would you like to generate a sample at './data/sample_training_data.csv'?"
                )
                if generate_sample:
                    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
                    generate_sample_data(sample_file)
                    print(f"Generated sample training data at: {sample_file}")

        analyzer = URLAnalyzer(args.model if os.path.exists(args.model) else None)

        if args.train:
            if not args.train_file:
                print("Error: Training requires a CSV file with URLs and labels")
                return

            try:
                # Load training data (CSV with 'url' and 'is_interesting' columns)
                train_df = pd.read_csv(args.train_file)
                urls = train_df["url"].tolist()
                labels = train_df["is_interesting"].tolist()

                analyzer.train(urls, labels)
                analyzer.save_model(args.model)
                print(f"Model trained and saved to {args.model}")
            except FileNotFoundError:
                print(f"Error: Training data file '{args.train_file}' not found.")
            except Exception as e:
                print(f"Error training model: {str(e)}")
        else:
            urls = []

            if args.file:
                if not args.quiet:
                    print(f"Reading URLs from {args.file}...")
                    sys.stdout.flush()
                urls = process_url_file(args.file)
                if not args.quiet:
                    print(f"Loaded {len(urls)} URLs.")
                    sys.stdout.flush()
            elif args.url:
                urls = [args.url]
            else:
                # Read from stdin
                if stdin_has_data:
                    if not args.quiet:
                        print("Reading URLs from stdin...")
                        sys.stdout.flush()
                    try:
                        stdin_wrapper = io.TextIOWrapper(
                            sys.stdin.buffer, encoding="utf-8", errors="replace"
                        )
                        urls = [line.strip() for line in stdin_wrapper if line.strip()]
                    except Exception as e:
                        print(f"Error reading from stdin: {e}", file=sys.stderr)
                        urls = []
                    if not args.quiet:
                        print(f"Loaded {len(urls)} URLs.")
                        sys.stdout.flush()

            if not urls:
                print("No URLs provided. Use -f, -u, or pipe URLs to stdin.")
                sys.stdout.flush()
                return

            try:
                if not args.quiet and len(urls) > 1000:
                    print(
                        f"Analyzing {len(urls)} URLs... This may take a while for large datasets."
                    )
                    sys.stdout.flush()

                ranked_urls = analyzer.rank_urls(urls, args.top)

                # Print results
                if not args.quiet:
                    print("\nAnalysis complete. Top potentially interesting URLs:")
                else:
                    print("\nTop potentially interesting URLs:")
                sys.stdout.flush()
                for i, (url, score) in enumerate(ranked_urls, 1):
                    print(f"{i}. [{score:.4f}] {url}")

            except NotFittedError:
                # Handle the case where the model is not trained
                display_training_instructions()
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Exiting.")
                sys.exit(0)
    except Exception as e:
        # Ensure error messages also use stderr and attempt UTF-8
        try:
            print(
                f"An error occurred: {str(e)}",
                file=sys.stderr,
                encoding="utf-8",
                errors="replace",
            )
        except TypeError:
            print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
