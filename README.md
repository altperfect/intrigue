# ML URL Analyzer for Security Testing

A tool that helps offensive security specialists to identify potentially interesting URLs from a large set of collected URLs.

## Overview

During security testing, tools like web crawlers, waybackurls, or urlfinder often produce large lists of URLs. Manually reviewing all these URLs is time-consuming. This tool helps prioritize the most potentially vulnerable or interesting URLs for security testing.

The tool uses machine learning to:
1. Analyze URL patterns
2. Identify paths, parameters, and file extensions associated with vulnerabilities
3. Rank URLs by their "interestingness" for security testing

## Features

- Takes input from files or stdin (pipe from other tools)
- Identifies potentially vulnerable endpoints
- Detects suspicious URL parameters
- Recognizes sensitive file extensions
- Finds admin interfaces, API endpoints, and authentication systems
- Ranks URLs by probability of security interest
- Provides output sorted by likelihood of vulnerability

## Installation

### Prerequisites

- Python 3.6+
- Required packages:
  - scikit-learn
  - pandas
  - joblib

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Analyze URLs from a file
python intrigue.py -f urls.txt

# Analyze a single URL
python intrigue.py -u https://example.com/admin/config.php

# Pipe output from other tools
cat urls.txt | python intrigue.py

# Get top 20 most interesting URLs
python intrigue.py -f urls.txt -n 20
```

### Training a custom model

The tool comes with a data generator to create training data:

```bash
# Generate training data
python intrigue.py --generate-sample --sample-size 200

# Train a model with your data
python intrigue.py --train --train-file data/my_training_data.csv
```

For better results, you can create your own training data by labeling URLs as interesting (1) or not interesting (0).

## How it works

The tool uses a machine learning pipeline with:
1. Text feature extraction (TF-IDF vectorizer on character n-grams)
2. Random Forest classifier

Key features that determine "interestingness":
- URL path components (admin, config, etc.)
- File extensions (php, asp, jsp, etc.)
- Parameter names (id, file, cmd, etc.)
- API endpoints
- Authentication-related paths
- Known dangerous patterns

## Example

Input:
```
https://example.com/products
https://example.com/admin/users
https://example.com/api/v1/users?id=1
https://example.com/login?redirect=https://attacker.com
https://example.com/download?file=../../../etc/passwd
```

Output:
```
Top potentially interesting URLs:
1. [0.9823] https://example.com/download?file=../../../etc/passwd
2. [0.9456] https://example.com/login?redirect=https://attacker.com
3. [0.8934] https://example.com/api/v1/users?id=1
4. [0.7612] https://example.com/admin/users
5. [0.2134] https://example.com/products
```