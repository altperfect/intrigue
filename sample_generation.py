import os
import random

import pandas as pd

from constants import BASE_DOMAINS, pattern_categories


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
            + (f"?page={random.randint(1, 5)}" if random.random() > 0.8 else "")
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
