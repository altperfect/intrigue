echo "Starting setup..."

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for uv
echo "[*] Checking for uv..."
if ! command_exists uv; then
    echo "[!] Error: 'uv' command not found."
    echo "[!] This script requires uv for package and environment management."
    echo "[!] To install uv, run 'pip install uv', or read more from: https://github.com/astral-sh/uv?tab=readme-ov-file#installation"
    return 1
fi

echo -e "\t[+] Found uv. Installing dependencies..."
INSTALL_CMD="uv pip install -r requirements.txt"
echo "[*] Running: $INSTALL_CMD"
$INSTALL_CMD
if [ $? -ne 0 ]; then
    echo "[!] Error: Dependency installation failed."
    return 1
fi
echo -e "\t[+] Dependencies installed successfully."

# Virtual env setup
echo "[*] Setting up virtual environment..."
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment in '$VENV_DIR'..."
    VENV_CMD="uv venv $VENV_DIR"
    $VENV_CMD
    if [ $? -ne 0 ]; then
        echo "[!] Error: Virtual environment creation failed."
        return 1
    fi
    echo -e "\t[+] Virtual environment created."
fi

# Check if already in an active virtual environment
ACTIVATED=false
if [ -z "$VIRTUAL_ENV" ]; then
    ACTIVATE_SCRIPT=""
    if [ -f "$VENV_DIR/bin/activate" ]; then
        ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
    fi

    if [ -n "$ACTIVATE_SCRIPT" ]; then
        echo "[*] Activating virtual environment: source $ACTIVATE_SCRIPT"
        source "$ACTIVATE_SCRIPT"
        ACTIVATED=true
    else
        echo "[!] Warning: Could not automatically determine activation script."
        echo "[!] Attempting common activation paths failed."
        echo "[!] If you are not already in the virtual environment, please activate it manually:"
        echo "  - Linux/macOS: source $VENV_DIR/bin/activate"
        echo "  - Windows CMD: $VENV_DIR\\Scripts\\activate.bat"
        echo "  - Windows PowerShell: $VENV_DIR\\Scripts\\Activate.ps1"
    fi
else
    echo -e "\t[+] Already in a virtual environment: $VENV_DIR"
    ACTIVATED=true
fi

# Exit if activation failed and wasn't already active
if ! $ACTIVATED && [ -z "$VIRTUAL_ENV" ]; then
    echo "[!] Error: Could not activate virtual environment. Setup cannot continue safely."
    return 1
fi

echo "[*] Exporting environment variables for this session..."
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8
echo -e "\t[+] PYTHONPATH set to: $(pwd)"
echo -e "\t[+] PYTHONIOENCODING set to: $PYTHONIOENCODING"

SAMPLE_SIZE=2000
echo "[*] Generating sample data ($SAMPLE_SIZE URLs) and training the model..."

# Determine python executable (prioritize 'python' within venv)
PYTHON_CMD=""
if command_exists python; then
    PYTHON_CMD="python"
    echo "[*] Using 'python' command."
elif command_exists python3; then
    PYTHON_CMD="python3"
    echo "[*] Using 'python3' command as fallback."
else
    echo "[!] Error: 'python' or 'python3' command not found within venv. Cannot run the setup command."
    return 1
fi

# Run the setup command to generate sample data and train the model
echo "[*] Running: $PYTHON_CMD src/intrigue.py --setup --sample-size $SAMPLE_SIZE --quiet"
SETUP_OUTPUT=$($PYTHON_CMD src/intrigue.py --setup --sample-size $SAMPLE_SIZE --quiet 2>&1)
SETUP_EXIT_CODE=$?

# Check if the setup command failed
if [ $SETUP_EXIT_CODE -ne 0 ]; then
    echo "[!] Error: The intrigue.py setup command failed with exit code $SETUP_EXIT_CODE."
    echo "[!] Output from the command:"
    echo "$SETUP_OUTPUT"
    return 1
else
    echo -e "\t[+] Model setup and training completed successfully."
fi

# Test the tool with example URLs
EXAMPLE_URLS_FILE="example_urls.txt"
echo "[*] Testing the tool with example URLs from $EXAMPLE_URLS_FILE file..."
if [ -f "$EXAMPLE_URLS_FILE" ]; then
    TEST_OUTPUT=$(cat "$EXAMPLE_URLS_FILE" | $PYTHON_CMD src/intrigue.py --quiet 2>&1)
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo -e "\t[+] Test completed successfully!"
    else
        echo "[!] Test failed with exit code: $TEST_EXIT_CODE"
        echo "[!] Error output:"
        echo "$TEST_OUTPUT"
        echo ""
        echo "[!] Please review the installation process and fix any errors before proceeding."
        echo "[!] You may need to check that the model was trained correctly and all dependencies are installed."
        return 1
    fi
else
    echo "[!] Warning: $EXAMPLE_URLS_FILE not found. Skipping test."
fi

echo ""
echo "-------------------------------------"
echo "Setup complete!"
echo " + The virtual environment is set up and dependencies are installed"
echo " + The model has been trained with $SAMPLE_SIZE sample URLs"
echo " + The tool has been tested with example URLs"
echo ""
echo "Environment variables PYTHONPATH and PYTHONIOENCODING have been set for this terminal session"
echo "If you open a new terminal, remember to activate the venv and potentially re-export PYTHONPATH: 'export PYTHONPATH=.'"
echo ""
echo "You can now analyze URLs. Example usage:"
echo "  > $PYTHON_CMD src/intrigue.py -f urls.txt"
echo "  > cat urls.txt | $PYTHON_CMD src/intrigue.py -n 20"
echo ""
echo "Advanced usage examples:"
echo "  # Generate new training data with a custom sample size"
echo "  $PYTHON_CMD src/intrigue.py --generate-sample --sample-size 30000"
echo ""
echo "  # Train a model with your own labeled data"
echo "  $PYTHON_CMD src/intrigue.py --train --train-file data/my_training_data.csv"
echo ""
echo "  # Regenerate training data and retrain the model in one step"
echo "  $PYTHON_CMD src/intrigue.py --setup --sample-size 10000"
echo ""
echo "  # View all available options"
echo "  $PYTHON_CMD src/intrigue.py --help"
echo "-------------------------------------"

return 0