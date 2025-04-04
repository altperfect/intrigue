# ML URL analyzer for ranking 

A tool that helps offensive security specialists to identify potentially interesting URLs from a large set of collected URLs.

## Overview

During security testing, you might end up with a large list of URLs compiled with crawlers or tools like waybackurls. Manually reviewing all these URLs is time-consuming. This tool helps prioritize the most potentially vulnerable or interesting URLs for security testing.

The tool uses machine learning to:
1. Analyze URL patterns
2. Identify paths, parameters, and file extensions associated with vulnerabilities
3. Rank URLs by their "interestingness" for security testing

## Installation

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

# Pipe output from other tools and save to a file
cat urls.txt | python intrigue.py -o interesting_urls.txt

# Get top 20 most interesting URLs
python intrigue.py -f urls.txt -n 20
```

### Training a custom model

The tool comes with a data generator to create training data in case you just want to get into the action:

```bash
# Generate training data
python intrigue.py --generate-sample --sample-size 2000

# Train a model with your data
python intrigue.py --train --train-file data/my_training_data.csv

# Or, run the --setup flag to generate the data and train the model in one step
python intrigue.py --setup

# ...it accepts --sample-size as well
python intrigue.py --setup --sample-size 30000
```

For better results, it's recommended to create your own training data by labeling URLs as interesting (1) or not interesting (0).

## Example

Input:
```
https://example.com/products
https://example.com/admin/users
https://example.com/api/v1/users?id=1
https://example.com/login?redirect=https://example.com/profile
https://example.com/download?file=/report.pdf
```

Output:
```
Top potentially interesting URLs:
1. [0.6305] https://example.com/login?redirect=https://example.com/profile
2. [0.5961] https://example.com/download?file=/report.pdf
3. [0.5094] https://example.com/admin/users
4. [0.2900] https://example.com/api/v1/users?id=1
5. [0.1159] https://example.com/products
```