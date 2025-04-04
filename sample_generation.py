import os
import random
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd

from constants import (
    ADMIN_PATH_SEGMENTS,
    API_PATH_SEGMENTS,
    AUTH_PATH_SEGMENTS,
    BASE_DOMAINS,
    BORING_PARAMS,
    COMMON_FILE_EXTENSIONS,
    COMMON_PATH_SEGMENTS,
    COMMON_VALUES,
    FILE_PATH_SEGMENTS,
    INTERESTING_FILE_EXTENSIONS,
    INTERESTING_PARAMS,
    URL_VALUE_PARAMS,
    generate_id,
    generate_random_string,
)


def _generate_path(max_depth: int = 4) -> str:
    """Generates a random URL path with varying depth.

    Args:
        max_depth: Maximum depth of the URL path. Defaults to 4.

    Returns:
        A string containing the generated URL path.
    """
    depth = random.randint(1, max_depth)
    segments = random.choices(
        ADMIN_PATH_SEGMENTS
        + API_PATH_SEGMENTS
        + AUTH_PATH_SEGMENTS
        + FILE_PATH_SEGMENTS
        + COMMON_PATH_SEGMENTS,
        k=depth,
    )
    path = "/" + "/".join(segments)

    if random.random() < 0.4:
        ext_type = random.choice(["interesting", "common"])
        if ext_type == "interesting":
            path += random.choice(INTERESTING_FILE_EXTENSIONS)
        else:
            path += random.choice(COMMON_FILE_EXTENSIONS)
    elif not path.endswith("/"):
        path += "/"

    return path


def _generate_query_string(max_params: int = 5) -> str:
    """Generates a random query string with interesting and boring params.

    Args:
        max_params: Maximum number of parameters to include in the query string. Defaults to 5.

    Returns:
        A URL-encoded query string containing randomly generated parameters.
    """
    num_params = random.randint(1, max_params)
    params = {}
    param_types_available = ["interesting", "boring"]

    for _ in range(num_params):
        param_type = random.choice(param_types_available)
        if param_type == "interesting" and INTERESTING_PARAMS:
            param_name = random.choice(INTERESTING_PARAMS)
        elif param_type == "boring" and BORING_PARAMS:
            param_name = random.choice(BORING_PARAMS)
        else:
            param_name = random.choice(INTERESTING_PARAMS + BORING_PARAMS)

        if param_name in URL_VALUE_PARAMS:
            value = (
                "https://"
                + random.choice(BASE_DOMAINS)
                + _generate_path(random.randint(1, 3))
            )
            if random.random() < 0.2:
                value += "?" + _generate_query_string(2)
        else:
            value_type = random.choice(["id", "string", "common", "combo"])
            if value_type == "id":
                value = generate_id(random.choice(["int", "uuid", "hex", "alnum"]))
            elif value_type == "string":
                value = generate_random_string(random.randint(5, 20))
            elif value_type == "combo" and COMMON_VALUES:
                value = random.choice(COMMON_VALUES) + generate_random_string(
                    random.randint(3, 8)
                )
            else:
                value = (
                    random.choice(COMMON_VALUES)
                    if COMMON_VALUES
                    else generate_random_string(5)
                )

        params[param_name] = value

    if random.random() < 0.1:
        inj_param = random.choice(list(params.keys()))
        if inj_param not in URL_VALUE_PARAMS:
            inj_type = random.choice(["pt", "sqli", "cmd"])
            if inj_type == "pt":
                params[inj_param] = random.choice(
                    ["../" * random.randint(1, 4) + "etc/passwd", "../../boot.ini"]
                )
            elif inj_type == "sqli" and isinstance(params[inj_param], str):
                params[inj_param] += random.choice(
                    ["' OR 1=1--", " UNION SELECT null--"]
                )
            elif inj_type == "cmd" and isinstance(params[inj_param], str):
                params[inj_param] += random.choice([";id", "|ls"])

    return urlencode(params)


def _determine_label(url: str) -> int:
    """Determine if a generated URL should be labeled interesting (1) or not (0)."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    params = parse_qs(parsed.query)
    filename = os.path.basename(path).lower()
    file_ext = os.path.splitext(filename)[1]
    path_parts = [p for p in path.split("/") if p]

    negative_paths = [
        "/support/",
        "/help/",
        "/faq/",
        "/blog/",
        "/news/",
        "/articles/",
        "/events/",
    ]
    negative_exts = [
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
    ]
    negative_docs = ["terms", "privacy", "license"]
    if any(np in path for np in negative_paths):
        return 0
    if file_ext in negative_exts:
        return 0
    if any(doc in filename for doc in negative_docs):
        return 0
    if query and all(p in BORING_PARAMS for p in params.keys()):
        if not any(
            p in path_parts
            for p in ADMIN_PATH_SEGMENTS
            + API_PATH_SEGMENTS
            + AUTH_PATH_SEGMENTS
            + FILE_PATH_SEGMENTS
        ):
            if not any(ext in file_ext for ext in INTERESTING_FILE_EXTENSIONS):
                return 0

    positive_paths = ADMIN_PATH_SEGMENTS + API_PATH_SEGMENTS + AUTH_PATH_SEGMENTS
    positive_file_paths = [
        "/upload",
        "/download",
        "/export",
        "/backup",
        "/import",
        "/include",
        "/config",
    ]
    if any(pp in path_parts for pp in positive_paths):
        return 1
    if any(pfp in path for pfp in positive_file_paths):
        return 1
    if any(p in params for p in INTERESTING_PARAMS if p not in BORING_PARAMS):
        return 1
    if (
        file_ext in INTERESTING_FILE_EXTENSIONS
        and file_ext not in COMMON_FILE_EXTENSIONS
    ):
        return 1
    if "../" in query or "%2e%2e" in query:
        return 1
    if any(p in URL_VALUE_PARAMS for p in params) and (
        any("://" in v[0] for v in params.values() if v)
    ):
        return 1

    return 0


def generate_sample_data(output_file: str, num_samples: int = 100) -> str:
    """
    Generate sample training data with labeled URLs, emphasizing security patterns.
    Uses dynamic generation with imported constants.

    Args:
        output_file: Path where the sample data will be saved
        num_samples: Number of sample URLs to generate

    Returns:
        Path to the generated sample file
    """
    all_urls = []

    total_needed = num_samples
    generated_urls = set()

    while len(all_urls) < total_needed:
        focus_type = random.choice(["interesting", "normal"])

        if focus_type == "interesting":
            interesting_focus = [
                "admin",
                "api",
                "auth",
                "file",
                "params",
                "redirect",
                "ext",
                "misc",
            ]
            focus = random.choice(interesting_focus)
            base_domain = random.choice(BASE_DOMAINS)
            path = ""
            query = ""

            depth = random.randint(1, 4)
            if focus == "admin" and ADMIN_PATH_SEGMENTS:
                segments = [random.choice(ADMIN_PATH_SEGMENTS)] + random.choices(
                    COMMON_PATH_SEGMENTS + FILE_PATH_SEGMENTS, k=depth - 1
                )
            elif focus == "api" and API_PATH_SEGMENTS:
                segments = [random.choice(API_PATH_SEGMENTS)] + random.choices(
                    COMMON_PATH_SEGMENTS + FILE_PATH_SEGMENTS + API_PATH_SEGMENTS,
                    k=depth - 1,
                )
            elif focus == "auth" and AUTH_PATH_SEGMENTS:
                segments = [random.choice(AUTH_PATH_SEGMENTS)] + random.choices(
                    COMMON_PATH_SEGMENTS + FILE_PATH_SEGMENTS, k=depth - 1
                )
            elif focus == "file" and FILE_PATH_SEGMENTS:
                segments = [random.choice(FILE_PATH_SEGMENTS)] + random.choices(
                    COMMON_PATH_SEGMENTS + FILE_PATH_SEGMENTS, k=depth - 1
                )
            else:
                segments = random.choices(
                    COMMON_PATH_SEGMENTS + FILE_PATH_SEGMENTS, k=depth
                )

            random.shuffle(segments)
            path = "/" + "/".join(segments)

            if focus == "ext" and INTERESTING_FILE_EXTENSIONS:
                path += random.choice(INTERESTING_FILE_EXTENSIONS)
            elif focus == "file" and random.random() < 0.6:
                path += random.choice(
                    INTERESTING_FILE_EXTENSIONS + COMMON_FILE_EXTENSIONS
                )
            elif random.random() < 0.2:
                path += random.choice(COMMON_FILE_EXTENSIONS)
            elif not path.endswith("/"):
                path += "/"

            if focus == "params" or focus == "redirect" or random.random() < 0.6:
                query = _generate_query_string(max_params=random.randint(1, 6))
                if focus == "redirect":
                    if not any(
                        p in query.lower()
                        for p in ["redirect", "url", "next", "back", "goto"]
                    ):
                        q_params = dict(
                            item.split("=") for item in query.split("&") if "=" in item
                        )
                        q_params[random.choice(["redirect_uri", "next", "r"])] = (
                            "https://" + random.choice(BASE_DOMAINS) + _generate_path(2)
                        )
                        query = urlencode(q_params)

        else:
            base_domain = random.choice(BASE_DOMAINS)
            path = _generate_path(random.randint(1, 3))
            query = ""

            if any(path.endswith(ext) for ext in INTERESTING_FILE_EXTENSIONS):
                path = os.path.splitext(path)[0] + random.choice(COMMON_FILE_EXTENSIONS)
            elif (
                not any(path.endswith(ext) for ext in COMMON_FILE_EXTENSIONS)
                and random.random() < 0.3
            ):
                path += random.choice(COMMON_FILE_EXTENSIONS)

            path_parts = path.strip("/").split("/")
            is_interesting_path = any(
                p in ADMIN_PATH_SEGMENTS + API_PATH_SEGMENTS + AUTH_PATH_SEGMENTS
                for p in path_parts
            )
            if is_interesting_path:
                new_segments = random.choices(
                    COMMON_PATH_SEGMENTS + FILE_PATH_SEGMENTS, k=len(path_parts)
                )
                path = "/" + "/".join(new_segments)
                if not path.endswith("/"):
                    path += "/"

            if random.random() < 0.3 and BORING_PARAMS:
                num_params = random.randint(1, 4)
                params = {}
                for _ in range(num_params):
                    param_name = random.choice(BORING_PARAMS)
                    value = (
                        generate_id()
                        if random.random() > 0.5
                        else random.choice(COMMON_VALUES)
                    )
                    params[param_name] = value
                query = urlencode(params)

        url = f"https://{base_domain}{path}"
        if query:
            url += f"?{query}"

        if url in generated_urls:
            continue
        generated_urls.add(url)

        label = _determine_label(url)
        all_urls.append((url, label))

    random.shuffle(all_urls)
    final_urls = all_urls[:num_samples]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(final_urls, columns=["url", "is_interesting"])
    df.to_csv(output_file, index=False)

    return output_file
