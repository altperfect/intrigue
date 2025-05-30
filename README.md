# intrigue: ML URL analyzer

A tool that helps offensive security specialists to identify potentially interesting URLs from a large set of collected URLs.

## Overview

During security testing, you often end up with a massive list of URLs from crawlers or tools like waybackurls. Manually reviewing all of them is tedious and slow. This tool uses machine learning to highlight the most interesting and potentially vulnerable URLs — so you can focus on what actually matters.

The tool uses machine learning to:
1. Analyze URL patterns
2. Identify paths, parameters, and file extensions associated with vulnerabilities
3. Rank URLs by their "interestingness" for security testing

## Quickstart

This will:
1. Install `uv` using the official installation script (assuming you're using Linux, MacOS or Git Bash)
2. Make the project's setup script executable
3. Run the script in `source` to preserve exported environment variables
4. Run the tool to analyze a test URL

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && \
chmod +x setup.sh && \
source setup.sh && \
python src/intrigue.py -u https://example.com/setup?complete=true -q
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Using setup.sh

You can run `setup.sh` to automate the installation process. The only prerequisite is to have `uv` installed.

```bash
# Install uv from pip
pip install uv

# Or from the installation script
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or explore other options from here: https://github.com/astral-sh/uv?tab=readme-ov-file#installation
```

```bash
chmod +x setup.sh

# Important: run with 'source' (not './setup.sh') in order to maintain exported variables
# The variables are needed to avoid potential issues when running the tool
source setup.sh
```

If you see `Setup complete!` message, then the tool is ready.

### Manual

```bash
# Install uv if you don't have it
pip install uv

# Install dependencies
uv pip install -r requirements.txt

# For development (includes pytest)
uv pip install -r requirements-dev.txt

# Create and activate a virtual environment
uv venv

# Export env variables to avoid potential errors and terminal issues
export PYTHONPATH=. && export PYTHONIOENCODING=utf-8
```

## Usage

### Basic usage

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

Using [example_urls.txt](https://github.com/altperfect/intrigue/blob/main/example_urls.txt) for test output.

```bash
$ cat example_urls.txt | python src/intrigue.py -q

Top potentially interesting URLs:
1. [0.5945] https://example.com/download?file=/report.pdf
2. [0.5200] https://example.com/admin/users
3. [0.4363] https://example.com/login?redirect=https://example.com/profile
4. [0.2687] https://example.com/api/v1/users?id=1
5. [0.0864] https://example.com/products
```

## Got something to add?

Feel free to contribute or open an issue in case you meet a bug or think of a new feature :)
