import argparse
import concurrent.futures
import functools
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

from constants import (
    IMAGE_EXTENSIONS,
    INTERESTING_PARAMS,
    INTERESTING_PATHS,
    LOCALIZATION_PATHS,
    LOW_PRIORITY_PATHS,
    STATIC_INDICATORS_FOR_FEATURE_EXTRACTION,
)
from sample_generation import generate_sample_data


class URLAnalyzer:
    """
    Analyzer for ranking URLs by security interest.

    Uses a TF-IDF vectorizer with character n-grams followed by a Random Forest
    classifier to identify potentially interesting URLs for security testing.
    """

    def __init__(self, model_path: Optional[str] = None, quiet: bool = False):
        """
        Initialize the URL analyzer with an optional pre-trained model.

        Args:
            model_path: Path to a saved model file. If None, a new model will be created.
            quiet: If True, suppress informational messages like model loading.
        """
        self.quiet = quiet
        self._features_cache = {}

        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                if not self.quiet:
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
            "GENERIC_LOGIN_PATH": 4,
            "DEFAULT": 30,
        }

        # Patterns for quick termination of clearly uninteresting URLs
        self.UNINTERESTING_PATTERNS = [
            # Static resources
            r"\.(jpg|jpeg|png|gif|svg|ico|css|woff|woff2|ttf|eot|map)(\?|$)",
            r"/(images|img|static|assets|styles|css|js|fonts|dist|build)/",
            # Analytics and tracking
            r"/(analytics|pixel|beacon|tracking|utm)/",
            r"(google-analytics|googletagmanager|facebook\.com/tr)",
            # Common tracking parameters
            r"[?&](utm_source|utm_medium|utm_campaign|utm_term|utm_content|fbclid|gclid)=",
            # Other clearly uninteresting paths
            r"/(sitemap\.xml|robots\.txt|favicon\.ico)(\?|$)",
        ]
        self.uninteresting_regex = re.compile(
            "|".join(self.UNINTERESTING_PATTERNS), re.IGNORECASE
        )

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
        early_term = self._check_early_termination(url)
        if early_term:
            return early_term

        if url in self._features_cache:
            return self._features_cache[url]

        try:
            parsed = self._parse_url(url)
            path = parsed.path.lower()
            query = parsed.query
            params = parse_qs(query)
        except ValueError:
            return "INVALID_URL"

        # Clean path from potential trailing garbage before extracting parts
        cleaned_path = path.rstrip("\"'`")

        features = []
        file_ext = os.path.splitext(cleaned_path)[1].lower()
        filename = os.path.basename(cleaned_path).lower()

        if file_ext in {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".css",
            ".js",
            ".ico",
            ".svg",
            ".woff",
            ".woff2",
            ".ttf",
        }:
            result = "STATIC_RESOURCE"
            self._features_cache[url] = result
            return result

        path_context = self._extract_path_context(path)
        features.extend([f for f in path_context.values() if f and isinstance(f, str)])

        tracking_features = self._extract_tracking_features(path, params)
        features.extend(tracking_features)

        if path_context["is_static_override"]:
            features.append("IS_STATIC_OVERRIDE")

        if any(lp in path for lp in LOCALIZATION_PATHS):
            features.append("LOCALIZATION_FILE")

        high_priority_features = self._extract_high_priority_features(
            path, params, tracking_features
        )
        features.extend(high_priority_features)

        file_features = self._extract_file_features(
            path, filename, file_ext, tracking_features, path_context
        )
        features.extend(file_features)

        auth_features = self._extract_auth_features(
            path, params, tracking_features, file_features
        )
        features.extend(auth_features)

        api_features = self._extract_api_features(path, tracking_features, path_context)
        features.extend(api_features)

        static_features = self._extract_static_features(
            path, file_ext, filename, tracking_features, path_context, file_features
        )
        features.extend(static_features)

        result = " ".join(features)
        self._features_cache[url] = result
        return result

    def _extract_path_context(self, path: str) -> Dict[str, str]:
        """Extract path context information to guide feature extraction."""
        static_indicators = STATIC_INDICATORS_FOR_FEATURE_EXTRACTION
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
        is_svg_icon_path = any(
            indicator in path for indicator in ["/icons/", "/svg/", "/avatar/"]
        )
        is_content_path = any(cp in path for cp in content_paths)
        is_support_path = any(sp in path for sp in support_paths)
        is_dev_docs_path = any(dp in path for dp in dev_docs_paths)
        is_survey_path = any(sp in path for sp in survey_paths)
        is_static_override = is_likely_static_path or is_svg_icon_path
        is_wp_admin_path = path.startswith("/wp-admin/")

        result = {
            "is_likely_static_path": is_likely_static_path,
            "is_svg_icon_path": is_svg_icon_path,
            "is_content_path": is_content_path,
            "is_support_path": "SUPPORT_PATH" if is_support_path else "",
            "is_dev_docs_path": is_dev_docs_path,
            "is_survey_path": (
                "SURVEY_PATH"
                if is_survey_path and "TRACKING_ENDPOINT" not in path
                else ""
            ),
            "is_static_override": is_static_override,
            "is_wp_admin_path": is_wp_admin_path,
        }

        # Low priority paths
        if any(lp in path for lp in LOW_PRIORITY_PATHS):
            result["low_priority_path"] = "LOW_PRIORITY_PATH"

        # Special paths
        if path.endswith("/certificate/private"):
            result["certificate_private"] = "CERTIFICATE_PRIVATE_PATH"

        if path.endswith("/managers"):
            result["managers_path"] = "MANAGERS_PATH"

        return result

    def _extract_tracking_features(
        self, path: str, params: Dict[str, List[str]]
    ) -> List[str]:
        """Extract tracking-related features from a URL."""
        features = []

        # Tracking/metrics/UTM detection
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
        has_promo_params = any(
            p.startswith("promo") or p.endswith("promo") for p in params.keys()
        )
        is_tracking = any(tp in path for tp in tracking_paths)
        tracking_in_value = False

        # Check parameter values for tracking indicators
        value_tracking_indicators = [
            "utm_",
            "_ga=",
            "mc=",
            "jsredir",
            "gclid",
            "fbclid",
        ]
        relevant_params_for_value_check = [
            "referer",
            "referrer",
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
            "state",
            "RelayState",
        ]

        for param_name, values in params.items():
            if param_name.lower() in relevant_params_for_value_check:
                for value in values:
                    if any(
                        indicator in value for indicator in value_tracking_indicators
                    ):
                        tracking_in_value = True
                        break
            if tracking_in_value:
                break

        if is_tracking:
            features.append("TRACKING_ENDPOINT")
        if has_utm_params:
            features.append("UTM_PARAMS")
        if has_promo_params:
            features.append("PROMO_PARAMS")
        if tracking_in_value:
            features.append("TRACKING_INDICATOR_IN_VALUE")

        return features

    def _extract_high_priority_features(
        self, path: str, params: Dict[str, List[str]], tracking_features: List[str]
    ) -> List[str]:
        """Extract high priority security features from URL parameters."""
        features = []
        is_tracking = any(
            f in ["TRACKING_ENDPOINT", "UTM_PARAMS", "PROMO_PARAMS"]
            for f in tracking_features
        )

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
                    and "TRACKING_ENDPOINT" not in tracking_features
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

        return features

    def _extract_file_features(
        self,
        path: str,
        filename: str,
        file_ext: str,
        tracking_features: List[str],
        path_context: Dict[str, str],
    ) -> List[str]:
        """Extract features related to file extensions and sensitive files."""
        features = []
        is_wp_admin_path = path_context.get("is_wp_admin_path", False)

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

        if file_ext in high_priority_exts:
            if "TRACKING_ENDPOINT" not in tracking_features:
                features.append(f"HIGH_PRIORITY_EXT:{file_ext}")
        elif file_ext in medium_priority_exts:
            if file_ext in [".json", ".xml"] and any(
                sp in path for sp in ["/static/", "/assets/"]
            ):
                features.append("STATIC_DATA_FILE")
            else:
                if "TRACKING_ENDPOINT" not in tracking_features:
                    features.append(f"MEDIUM_PRIORITY_EXT:{file_ext}")
        elif file_ext in low_priority_exts:
            if "TRACKING_ENDPOINT" not in tracking_features:
                features.append(f"LOW_PRIORITY_EXT:{file_ext}")

        if filename in sensitive_filenames:
            if "TRACKING_ENDPOINT" not in tracking_features:
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
                "../" in path
                or "%2e%2e%2f" in path.lower()
                or "..\\" in path
                or "%2e%2e\\" in path.lower()
            ):
                features.append("PATH_TRAVERSAL_PATTERN")

        if is_wp_admin_path and filename in wp_admin_sensitive_files:
            if "TRACKING_ENDPOINT" not in tracking_features:
                features.append("WP_ADMIN_SENSITIVE_FILE")

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

        if file_ext in IMAGE_EXTENSIONS:
            features.append("IMAGE_FILE")

        return features

    def _extract_auth_features(
        self,
        path: str,
        params: Dict[str, List[str]],
        tracking_features: List[str],
        file_features: List[str],
    ) -> List[str]:
        """Extract authentication and OAuth flow related features."""
        features = []

        oidc_oauth_paths = [
            "/connect/authorize",
            "/oauth/authorize",
            "/oidc",
            "/saml",
            "/login",
            "/signin",
            "/auth",
            "/account/login",
            "/user/login",
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
        if "TRACKING_ENDPOINT" not in tracking_features and (
            any(p in path for p in oidc_oauth_paths)
            or any(p in params for p in oidc_oauth_params)
        ):
            features.append("OAUTH_SAML_FLOW")
            is_oidc_flow = True

            # Check for GENERIC_LOGIN_PATH: OAuth flow is true, but no higher priority features are present
            high_priority_features = [
                "SSRF_REDIRECT_URL_IN_PARAM",
                "SSRF_REDIRECT_CANDIDATE",
                "DANGEROUS_PARAM",
                "PATH_TRAVERSAL_PATTERN",
                "SENSITIVE_FILENAME",
                "HIGH_PRIORITY_EXT",
                "WP_CONFIG_FILE",
                "WP_ADMIN_SENSITIVE_FILE",
                "WP_ADMIN_AREA",
            ]

            is_high_priority = any(
                f in file_features or any(hpf in f for hpf in high_priority_features)
                for f in file_features
            )

            # Check if it's one of the specific common login paths
            is_common_login_path = any(
                p == path
                for p in [
                    "/login",
                    "/signin",
                    "/auth",
                    "/account/login",
                    "/user/login",
                    # API version patterns
                    "/v1/login",
                    "/v2/login",
                    "/v3/login",
                    "/v1/account/login",
                    "/v2/account/login",
                    "/v3/account/login",
                    "/v1/auth",
                    "/v2/auth",
                    "/v3/auth",
                    # Auth directory patterns
                    "/connect/login",
                    "/oauth/login",
                    "/oidc/login",
                    "/saml/login",
                    # File extension patterns
                    "/login.aspx",
                    "/signin.aspx",
                    "/Authenticate.aspx",
                ]
            )

            if is_common_login_path and not is_high_priority:
                features.append("GENERIC_LOGIN_PATH")

        # .well-known directory checks
        if "TRACKING_ENDPOINT" not in tracking_features and path.startswith(
            "/.well-known/"
        ):
            filename = os.path.basename(path.rstrip("/")).lower()
            if filename == "openid-configuration":
                if not is_oidc_flow:
                    features.append("WELL_KNOWN_OPENID_CONFIG")
            elif filename == "security.txt":
                features.append("WELL_KNOWN_SECURITY_TXT")
            else:
                features.append("WELL_KNOWN_STATIC")

        return features

    def _extract_api_features(
        self, path: str, tracking_features: List[str], path_context: Dict[str, str]
    ) -> List[str]:
        """Extract API and admin-related features."""
        features = []
        is_tracking_present = (
            "TRACKING_ENDPOINT" in tracking_features
            or "UTM_PARAMS" in tracking_features
            or "PROMO_PARAMS" in tracking_features
        )
        is_dev_docs_path = path_context.get("is_dev_docs_path", False)
        is_likely_static_path = path_context.get("is_likely_static_path", False)
        is_wp_admin_path = path_context.get("is_wp_admin_path", False)

        # Check for JS files
        file_ext = os.path.splitext(path)[1].lower()
        if not is_tracking_present and file_ext == ".js":
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
                filename = os.path.basename(path).lower()
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

        if any(ap in path for ap in admin_paths) and is_likely_static_path:
            features.append("STATIC_ADMIN_INDICATOR")
        elif is_wp_admin_path:
            if not is_tracking_present:
                features.append("WP_ADMIN_AREA")
        elif any(ap in path for ap in admin_paths):
            if not is_tracking_present:
                features.append("ADMIN_AREA")

        if "TRACKING_ENDPOINT" not in tracking_features and (
            "/api/" in path or "/rest/" in path or "/graphql" in path
        ):
            features.append("API_ENDPOINT")
            sensitive_api_keywords = [
                "token",
                "key",
                "secret",
                "auth",
                "session",
                "script",
                "config",
                "admin",
                "debug",
                "internal",
                "private",
            ]
            if any(keyword in path for keyword in sensitive_api_keywords):
                features.append("API_SENSITIVE_KEYWORD")

        # Developer documentation
        if is_dev_docs_path and not is_tracking_present:
            if any(kw in path for kw in ["/com/", "/objects/", "/tools/"]):
                features.append("DEV_DOCS_TOOLS_PATH")
            else:
                features.append("DEV_DOCS_GENERIC_PATH")

        return features

    def _extract_static_features(
        self,
        path: str,
        file_ext: str,
        filename: str,
        tracking_features: List[str],
        path_context: Dict[str, str],
        file_features: List[str],
    ) -> List[str]:
        """Extract static resource and content page features."""
        features = []
        is_tracking_present = (
            "TRACKING_ENDPOINT" in tracking_features
            or "UTM_PARAMS" in tracking_features
            or "PROMO_PARAMS" in tracking_features
        )
        is_static_override = path_context.get("is_static_override", False)
        is_content_path = path_context.get("is_content_path", False)
        is_dev_docs_path = path_context.get("is_dev_docs_path", False)

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

        # Final static/content page identification
        if (
            not is_tracking_present
            and "COMMON_JS_FILE" not in file_features
            and "HASHED_JS_FILE" not in file_features
            and "ROBOTS_TXT" not in file_features
            and "COMMON_DOC_FILENAME" not in file_features
        ):
            if "IS_STATIC_OVERRIDE" in path_context.values() and file_ext == ".svg":
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

        return features

    def _quick_filter(self, urls: List[str]) -> List[str]:
        """
        Perform quick initial filtering on URLs to identify potentially interesting candidates.

        Args:
            urls: List of URLs to filter

        Returns:
            List of URLs that passed the quick filter
        """
        # Use a set for fast lookups when we have lots of URLs
        interesting_indicators_set = {
            "/api/",
            "/admin/",
            "/auth/",
            "/login/",
            "/config",
            "password",
            "token",
            "key",
            "secret",
            "/.well-known/",
            "/oauth",
            "/debug",
            "/upload",
            "/download",
            "/file",
        }

        candidates = []
        batch_size = 10000  # Process URLs in batches to reduce memory usage

        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]  # noqa: E203
            batch_candidates = []

            for url in batch:
                if self.uninteresting_regex.search(url):
                    continue

                url_lower = url.lower()

                if any(
                    indicator in url_lower for indicator in interesting_indicators_set
                ):
                    batch_candidates.append(url)
                    continue

                if "?" in url_lower and "=" in url_lower:
                    batch_candidates.append(url)
                    continue

                if any(
                    ext in url_lower
                    for ext in [".jpg", ".jpeg", ".png", ".gif", ".css", ".js"]
                ):
                    continue

                batch_candidates.append(url)

            candidates.extend(batch_candidates)

        return candidates

    def _process_url_batch(self, batch: List[str]) -> List[Tuple[str, float]]:
        """
        Process a batch of URLs to extract features and get prediction scores.

        Args:
            batch: List of URLs to process

        Returns:
            List of (url, score) tuples
        """
        batch_enhanced = []
        for url in batch:
            features = self._extract_security_features(url)
            enhanced_url = f"{url} {features}"
            batch_enhanced.append(enhanced_url)

        batch_probs = self.model.predict_proba(batch_enhanced)

        return [(url, prob[1]) for url, prob in zip(batch, batch_probs)]

    def rank_urls(self, urls: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Rank URLs by their security interestingness.

        Args:
            urls: List of URLs to analyze
            top_n: Number of top results to return

        Returns:
            List of (url, score) tuples sorted by score in descending order
        """
        if not urls:
            return []

        unique_urls = list(dict.fromkeys(urls))

        if not self.quiet:
            print(f"Processing {len(unique_urls)} unique URLs out of {len(urls)} total")

        potential_candidates = self._quick_filter(unique_urls)

        if not self.quiet:
            print(
                f"Quick filter reduced {len(unique_urls)} URLs to {len(potential_candidates)} candidates"
            )

        SMALL_DATASET_THRESHOLD = 100
        MAX_WORKERS = min(32, os.cpu_count() or 4)

        try:
            if len(potential_candidates) <= SMALL_DATASET_THRESHOLD:
                all_scores = self._process_url_batch(potential_candidates)
            else:
                all_scores = []

                # Split processing into smaller chunks to avoid memory issues
                chunk_size = min(
                    5000, max(100, len(potential_candidates) // (MAX_WORKERS * 2))
                )
                batches = [
                    potential_candidates[i : i + chunk_size]  # noqa: E203
                    for i in range(0, len(potential_candidates), chunk_size)
                ]

                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=MAX_WORKERS
                    ) as executor:
                        futures = []
                        for batch in batches:
                            future = executor.submit(self._process_url_batch, batch)
                            futures.append(future)

                        for future in concurrent.futures.as_completed(futures):
                            try:
                                batch_scores = future.result()
                                all_scores.extend(batch_scores)
                            except Exception as exc:
                                print(f"Batch processing generated an exception: {exc}")
                except Exception as e:
                    # Fall back to sequential processing if parallel processing fails
                    if not self.quiet:
                        print(
                            f"Warning: Parallel processing failed ({str(e)}). Falling back to sequential processing."
                        )
                    all_scores = []
                    for batch in batches:
                        batch_scores = self._process_url_batch(batch)
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
            "UNUSUAL_JS_FILE": 0.01,
            "API_ENDPOINT": 0.05,
            "API_SENSITIVE_KEYWORD": 0.15,
            "DEV_DOCS_TOOLS_PATH": 0.16,
            "DEV_DOCS_GENERIC_PATH": 0.05,
            "WELL_KNOWN_OPENID_CONFIG": 0.03,
            "WELL_KNOWN_SECURITY_TXT": -0.75,
            # low priority / context
            "FILE_ACCESS_PATH": 0.25,
            "LOW_PRIORITY_EXT": 0.01,
            "HAS_PARAMS": 0.01,
            "PARAM_COUNT": 0.005,
            "SIMPLE_QUERY_PARAM": 0.02,
            # penalties
            "LOW_PRIORITY_PATH": -0.95,
            "TRACKING_ENDPOINT": -1.0,
            "UTM_PARAMS": -0.70,
            "SURVEY_PATH": -0.70,
            "BORING_PARAMS": -0.10,
            "STATIC_RESOURCE": -0.80,
            "PLAIN_HTML_FILE": -0.60,
            "ROBOTS_TXT": -0.80,
            "COMMON_DOC_FILENAME": -0.80,
            "CONTENT_PAGE": -0.45,
            "COMMON_JS_FILE": -0.70,
            "HASHED_JS_FILE": -0.40,
            "SUPPORT_PATH": -0.30,
            "IS_STATIC_OVERRIDE": -0.70,
            "DOCS_STATIC_JS": -0.35,
            "CERTIFICATE_PRIVATE_PATH": -0.75,
            "MANAGERS_PATH": -0.45,
            "ICON_SVG_STATIC": -0.70,
            "HTML_DOC_IN_DEV_PATH": -0.25,
            "STATIC_DATA_FILE": -0.15,
            "WELL_KNOWN_STATIC": -0.35,
            "PROMO_PARAMS": -0.75,
            "TRACKING_INDICATOR_IN_VALUE": -0.75,
            "GENERIC_LOGIN_PATH": -0.20,
            "INVALID_URL": -1.0,
            "LOCALIZATION_FILE": -0.80,
            "IMAGE_FILE": -0.90,
        }

        MIN_SCORE = 0.01

        adjusted_scores = []
        for url, initial_score in url_scores:
            # Use cached features if available, otherwise extract them
            features_str = self._features_cache.get(
                url
            ) or self._extract_security_features(url)
            features = features_str.split()

            # Use cached URL parsing
            parsed = self._parse_url(url)

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
            is_api_sensitive = "API_SENSITIVE_KEYWORD" in features
            is_file_access = "FILE_ACCESS_PATH" in features
            is_path_traversal = "PATH_TRAVERSAL_PATTERN" in features
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
                            "PROMO_PARAMS",
                            "TRACKING_INDICATOR_IN_VALUE",
                            "IMAGE_FILE",
                            "STATIC_RESOURCE",
                            "COMMON_JS_FILE",
                            "LOW_PRIORITY_PATH",
                        ]:
                            primary_penalty_feature = base_feature
                    if weight >= 0.18:
                        found_high_priority_feature = True

            # scoring logic
            if primary_penalty_feature:
                penalty = feature_weights.get(primary_penalty_feature, -0.7)
                final_score = base_score * 0.1 + penalty * 1.1

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
            elif is_file_access and is_path_traversal:
                final_score = min(
                    0.90,
                    0.45
                    + base_score * 0.15
                    + positive_feature_score_sum * 0.93
                    + negative_feature_score_sum,
                )
            elif is_file_access:
                calculated_score = (
                    0.40
                    + base_score * 0.20
                    + positive_feature_score_sum * 0.80
                    + negative_feature_score_sum
                )

                try:
                    file_ext = os.path.splitext(parsed.path)[1].lower()
                    files_to_penalize = {
                        ".js",
                        ".css",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".svg",
                        ".ico",
                        ".woff",
                        ".woff2",
                        ".ttf",
                        ".eot",
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".xlsx",
                        ".xls",
                        ".pptx",
                        ".ppt",
                        ".zip",
                        ".tar",
                        ".gz",
                        ".mp4",
                        ".webm",
                        ".ogg",
                        ".mp3",
                        ".wav",
                        ".xml",
                        ".txt",
                        ".log",
                        ".json",
                        ".csv",
                    }
                    # Use 'features' list directly to check for path traversal feature
                    if (
                        file_ext in files_to_penalize
                        and "PATH_TRAVERSAL_PATTERN" not in features
                    ):
                        calculated_score = base_score * 0.2
                except Exception:
                    # Ignore errors in this secondary check
                    pass

                final_score = min(0.85, calculated_score)

            elif is_redirect and is_file_access:  # Less likely now but kept as fallback
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
                    0.45
                    + base_score * 0.2
                    + positive_feature_score_sum * 0.65
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
                    0.40,
                    0.15
                    + base_score * 0.2
                    + positive_feature_score_sum * 0.55
                    + negative_feature_score_sum,
                )
            elif is_api_sensitive:
                final_score = min(
                    0.55,
                    0.25
                    + base_score * 0.25
                    + positive_feature_score_sum * 0.65
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
            final_score = max(MIN_SCORE, min(0.95, final_score))
            adjusted_scores.append((url, final_score))

        ranked = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)

        # Diversity logic
        result = []
        pattern_counts = defaultdict(int)
        feature_counts = defaultdict(int)
        # Track path structural patterns to penalize repetition
        path_structure_counts = defaultdict(int)

        for url, score in ranked:
            parsed = urlparse(url)
            domain = parsed.netloc
            path_lower = parsed.path.lower()
            path_parts = [p for p in path_lower.split("/") if p]
            features_str = self._extract_security_features(url)
            features = features_str.split()

            primary_feature_key = "DEFAULT"
            # Prioritize GENERIC_LOGIN_PATH for its specific counter if present
            if "GENERIC_LOGIN_PATH" in features:
                primary_feature_key = "GENERIC_LOGIN_PATH"
            else:
                for key in self.MAX_REPETITIONS:
                    if (
                        key in features
                        and key != "DEFAULT"
                        and key != "GENERIC_LOGIN_PATH"
                    ):
                        primary_feature_key = key
                        break

            adjusted_score = score
            if primary_feature_key == "DEFAULT":
                adjusted_score -= 0.10
                adjusted_score = max(MIN_SCORE, adjusted_score)

            simplified_path_parts = []

            variable_pattern = re.compile(
                r"^[a-f0-9]{8,}$|"  # Hex string (8+ chars)
                r"^\d+$|"  # Number (any length)
                r"^[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$|"  # UUID
                # Concatenated string for extensions to fix line length
                r"\.(php|aspx?|jsp|cfm|cgi|pl|py|rb|sh|exe|dll|pdf|docx?|xlsx?|pptx?|"
                r"zip|tar|gz|rar|sql|bak|log|txt|csv|json|xml|yaml|yml|env|config|conf|"
                r"pem|key|p12|"
                r"js|css|png|jpe?g|gif|svg|ico|woff2?|ttf|eot|html?)$"  # Common extensions
            )

            for part in path_parts:
                if variable_pattern.search(part):
                    break
                simplified_path_parts.append(part)

            num_simplified_parts = len(simplified_path_parts)
            if num_simplified_parts == 0:
                path_base = "ROOT"
            elif num_simplified_parts == 1:
                path_base = simplified_path_parts[0]
            elif num_simplified_parts == 2:
                path_base = ":".join(simplified_path_parts)
            else:
                path_base = ":".join(simplified_path_parts[:3])

            path_pattern_key = f"{domain}:{path_base}"

            # Create a more generic structural pattern key ignoring domain
            # This helps detect repetitive paths across different domains
            structure_key = ":".join(
                [p if len(p) < 20 else f"*{len(p)}" for p in simplified_path_parts]
            )

            # Progressive penalty for repeated structures
            structure_count = path_structure_counts.get(structure_key, 0)
            if structure_count > 0:
                # Apply increasingly stronger penalties for each repetition
                repetition_penalty = min(0.15 * structure_count, 0.60)
                adjusted_score -= repetition_penalty
                adjusted_score = max(MIN_SCORE, adjusted_score)

            max_feature_allowed = self.MAX_REPETITIONS.get(
                primary_feature_key, self.MAX_REPETITIONS["DEFAULT"]
            )

            max_path_allowed = 1

            current_feature_count = feature_counts.get(primary_feature_key, 0)
            current_path_count = pattern_counts.get(path_pattern_key, 0)

            if (
                current_feature_count >= max_feature_allowed
                or current_path_count >= max_path_allowed
            ):
                continue

            result.append((url, adjusted_score))
            feature_counts[primary_feature_key] = current_feature_count + 1
            pattern_counts[path_pattern_key] = current_path_count + 1
            # Increment the structure counter
            path_structure_counts[structure_key] = structure_count + 1

        return sorted(result, key=lambda x: x[1], reverse=True)

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            model_path: Path where the model will be saved
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)

    @functools.lru_cache(maxsize=10000)
    def _parse_url(self, url: str):
        """
        Parse URL with caching for better performance.

        Args:
            url: URL to parse

        Returns:
            Parsed URL object
        """
        return urlparse(url)

    def _check_early_termination(self, url: str) -> Optional[str]:
        """
        Check if a URL should be terminated early as uninteresting.

        Args:
            url: URL to check

        Returns:
            Feature string for uninteresting URLs, None if URL should be analyzed
        """
        # Fast check for common static resources and tracking URLs
        if self.uninteresting_regex.search(url):
            return "EARLY_TERMINATED STATIC_RESOURCE"

        return None


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


def contains_cyrillic(text: str) -> bool:
    """
    Check if the provided text contains Cyrillic characters.

    Args:
        text: String to check for Cyrillic characters

    Returns:
        True if Cyrillic characters are found, False otherwise
    """
    cyrillic_pattern = re.compile(r"[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF]")
    return bool(cyrillic_pattern.search(text))


def display_cyrillic_warning() -> None:
    """Display warning message for files containing Cyrillic characters."""
    print("\nWARNING: Provided input contains Cyrillic characters.")
    print("If you see broken output, try running:")
    print("    export PYTHONIOENCODING=utf-8")
    print("in your terminal before running this script.\n")


def process_url_file(file_path: str, quiet: bool = False) -> List[str]:
    """
    Read URLs from a file, one per line.

    Args:
        file_path: Path to the file containing URLs
        quiet: If True, suppress informational messages

    Returns:
        List of URLs read from the file
    """
    URL_READ_PROGRESS_INTERVAL = 100000
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB in bytes

    try:
        file_size = os.path.getsize(file_path)

        def read_urls_from_file(
            file_path: str, show_progress: bool = False
        ) -> Tuple[List[str], bool]:
            """
            Read URLs from a file and check for Cyrillic characters.

            Args:
                file_path: Path to the file containing URLs
                show_progress: Whether to show progress for large files

            Returns:
                Tuple of (list of URLs, whether file contains Cyrillic characters)
            """
            urls = []
            has_cyrillic = False
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        urls.append(line)
                        if not has_cyrillic and contains_cyrillic(line):
                            has_cyrillic = True

                    if show_progress and len(urls) % URL_READ_PROGRESS_INTERVAL == 0:
                        print(f"Read {len(urls)} URLs so far...")

            return urls, has_cyrillic

        # Use buffered reading for large files
        if file_size > LARGE_FILE_THRESHOLD:
            print("Large dataset provided. This may take a while...")
            urls, has_cyrillic = read_urls_from_file(file_path, show_progress=True)
        else:
            urls, has_cyrillic = read_urls_from_file(file_path)

        if has_cyrillic and not quiet:
            display_cyrillic_warning()

        return urls
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []


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
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except Exception as e:
        print(f"Warning: Could not configure stdout for UTF-8: {e}", file=sys.stderr)


def read_from_stdin() -> Tuple[List[str], bool]:
    """
    Read URLs from stdin.

    Returns:
        Tuple of (list of URLs, whether stdin contains Cyrillic characters)
    """
    urls = []
    has_cyrillic = False
    try:
        stdin_wrapper = io.TextIOWrapper(
            sys.stdin.buffer, encoding="utf-8", errors="replace"
        )
        for line in stdin_wrapper:
            line_stripped = line.strip()
            if line_stripped:
                urls.append(line_stripped)
                if not has_cyrillic and contains_cyrillic(line_stripped):
                    has_cyrillic = True
    except Exception as e:
        print(f"Error reading from stdin: {e}", file=sys.stderr)

    return urls, has_cyrillic


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
            default=2000,
            help="Number of sample URLs to generate (when using --generate-sample)",
        )
        parser.add_argument(
            "--sample-output",
            default="data/sample_training_data.csv",
            help="Output file for sample data (when using --generate-sample)",
        )
        parser.add_argument(
            "--setup",
            action="store_true",
            help="Generate sample data and train the model in one step (accepts --sample-size)",
        )
        parser.add_argument(
            "-q", "--quiet", action="store_true", help="Suppress progress messages"
        )
        parser.add_argument(
            "-o", "--outfile", help="Save results to a file instead of stdout"
        )

        args = parser.parse_args()

        if not args.quiet:
            setup_utf8_stdout()

        if args.setup:
            print("Setting up with sample data and training the model...")
            os.makedirs("data", exist_ok=True)
            os.makedirs("models", exist_ok=True)

            sample_file = generate_sample_data(
                "data/sample_training_data.csv", num_samples=args.sample_size
            )
            print(
                f"Generated sample training data with {args.sample_size} URLs at: {sample_file}"
            )

            analyzer = URLAnalyzer(args.model, quiet=args.quiet)
            train_df = pd.read_csv(sample_file)
            urls = train_df["url"].tolist()
            labels = train_df["is_interesting"].tolist()

            analyzer.train(urls, labels)
            analyzer.save_model(args.model)
            print(f"Model trained and saved to {args.model}")
            print("\nSetup complete! To analyze URLs, run:")
            print("python intrigue.py -f path/to/urls.txt")
            return

        if args.generate_sample:
            sample_file = generate_sample_data(args.sample_output, args.sample_size)
            print(
                f"Generated sample training data with {args.sample_size} URLs at: {sample_file}\n"
                "To train the model with this data, run:\n"
                f"python intrigue.py --train --train-file {sample_file}"
            )
            return

        stdin_has_data = not sys.stdin.isatty()

        sample_file = "data/sample_training_data.csv"
        if not os.path.exists(sample_file) and not args.train:
            # If we're using stdin for URLs, generate sample data without prompting
            if stdin_has_data:
                print("No training data found. Generating sample data for future use.")
                os.makedirs(os.path.dirname(sample_file), exist_ok=True)
                generate_sample_data(sample_file)
                print(f"Generated sample training data at: {sample_file}")
            else:
                generate_sample = prompt_yes_no(
                    "No training data found. Would you like to generate a sample at './data/sample_training_data.csv'?"
                )
                if generate_sample:
                    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
                    generate_sample_data(sample_file)
                    print(f"Generated sample training data at: {sample_file}")

        analyzer = URLAnalyzer(args.model, quiet=args.quiet)

        if args.train:
            if not args.train_file:
                print("Error: Training requires a CSV file with URLs and labels")
                return

            try:
                train_df = pd.read_csv(args.train_file)
                urls = train_df["url"].tolist()
                labels = train_df["is_interesting"].tolist()

                analyzer.train(urls, labels)
                analyzer.save_model(args.model)
                print(
                    f"""Model trained and saved to {args.model}\n
To analyze URLs, run: python intrigue.py -f path/to/urls.txt"""
                )
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
                urls = process_url_file(args.file, quiet=args.quiet)
                if not args.quiet:
                    print(f"Loaded {len(urls)} URLs.")
                    sys.stdout.flush()
            elif args.url:
                urls = [args.url]
                if contains_cyrillic(args.url) and not args.quiet:
                    display_cyrillic_warning()
            else:
                if stdin_has_data:
                    if not args.quiet:
                        print("Reading URLs from stdin...")
                        sys.stdout.flush()

                    urls, has_cyrillic = read_from_stdin()

                    if has_cyrillic and not args.quiet:
                        display_cyrillic_warning()

                    if not args.quiet:
                        print(f"Loaded {len(urls)} URLs.")
                        sys.stdout.flush()

            if not urls:
                print("No URLs provided. Use -f, -u, or pipe URLs to stdin.")
                sys.stdout.flush()
                return

            try:
                if not args.quiet and len(urls) > 1000:
                    print("Analyzing...")
                    sys.stdout.flush()

                ranked_urls = analyzer.rank_urls(urls, args.top)

                if args.url:
                    header = "Analysis result for the provided URL:"
                    url, score = ranked_urls[0]
                    output_lines = [f"[{score:.4f}] {url}"]
                else:
                    header = "Top potentially interesting URLs:"
                    output_lines = []
                    for i, (url, score) in enumerate(ranked_urls, 1):
                        output_lines.append(f"{i}. [{score:.4f}] {url}")

                if args.outfile:
                    try:
                        outfile_dir = os.path.dirname(args.outfile)
                        if outfile_dir:
                            os.makedirs(outfile_dir, exist_ok=True)
                        with open(args.outfile, "w", encoding="utf-8") as f:
                            f.write(header + "\n")
                            for line in output_lines:
                                f.write(line + "\n")
                        if not args.quiet:
                            print(f"Results saved to {args.outfile}")
                    except IOError as e:
                        print(
                            f"Error writing results to file {args.outfile}: {e}",
                            file=sys.stderr,
                        )
                else:
                    print(f"\n{header}")
                    sys.stdout.flush()
                    for line in output_lines:
                        print(line)

            except NotFittedError:
                display_training_instructions()
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Exiting.")
                sys.exit(0)
    except Exception as e:
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
