echo "Starting setup..."

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for uv
echo "[*] Checking for uv..."
UV_PATH="$(command -v uv)"
if [ -z "$UV_PATH" ]; then
    echo "[!] Error: 'uv' command not found."
    echo "[!] This script requires uv for package and environment management."
    echo ""
    echo "To install uv, you can use the following commands:"
    echo "  - (recommended) pip install uv"
    echo "  - (linux/macos) curl -fsSL https://get.uv.dev | sh"
    echo "  - (windows) powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    echo "    Alternatively, refer to the official documentation of uv: https://github.com/astral-sh/uv?tab=readme-ov-file#installation"
    return 1
fi
echo -e "\t[+] Using uv found at: $UV_PATH"

# Virtual env setup
echo "[*] Setting up virtual environment..."
VENV_DIR=".venv"

# Determine OS type
OS_TYPE=$(uname -s)
EXPECTED_ACTIVATE=""
INCOMPATIBLE_ACTIVATE=""

case "$OS_TYPE" in
    Linux*|Darwin*)
        # Linux or macOS
        EXPECTED_ACTIVATE="$VENV_DIR/bin/activate"
        INCOMPATIBLE_ACTIVATE="$VENV_DIR/Scripts/activate"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # Windows Git Bash, MSYS, Cygwin
        EXPECTED_ACTIVATE="$VENV_DIR/Scripts/activate"
        INCOMPATIBLE_ACTIVATE="$VENV_DIR/bin/activate"
        ;;
    *)
        # Unknown OS, default to Linux/macOS style but warn
        echo "[!] Warning: Unknown OS type. Assuming Linux/macOS activation path."
        EXPECTED_ACTIVATE="$VENV_DIR/bin/activate"
        INCOMPATIBLE_ACTIVATE="$VENV_DIR/Scripts/activate"
        ;;
esac

# Check if venv already exists and handle if it is incompatible
NEEDS_RECREATION=false
if [ -d "$VENV_DIR" ]; then
    # If the incompatible script path exists OR the expected script path is missing, flag for recreation
    if [ -f "$INCOMPATIBLE_ACTIVATE" ] || [ ! -f "$EXPECTED_ACTIVATE" ]; then
         if [ ! -f "$EXPECTED_ACTIVATE" ] && [ ! -f "$INCOMPATIBLE_ACTIVATE" ]; then
             echo "[!] Existing virtual environment seems incomplete or corrupted (missing activation script)."
         else
             echo "[!] Detected potentially incompatible virtual environment structure (found '$INCOMPATIBLE_ACTIVATE' or missing '$EXPECTED_ACTIVATE' for OS type '$OS_TYPE')."
         fi
        NEEDS_RECREATION=true
    fi

    if $NEEDS_RECREATION; then
        echo "[*] Removing incompatible or corrupted virtual environment: $VENV_DIR"
        rm -rf "$VENV_DIR"
        if [ -d "$VENV_DIR" ]; then
            echo "[!] Error: Failed to remove existing virtual environment directory. Please remove it manually: rm -rf $VENV_DIR"
            return 1
        fi
        echo -e "\t[+] Old virtual environment removed."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment..."
    
    "$UV_PATH" venv "$VENV_DIR"
    VENV_EXIT_CODE=$?

    if [ $VENV_EXIT_CODE -ne 0 ]; then
        echo "[!] Error: Virtual environment creation failed with exit code $VENV_EXIT_CODE."
        return 1
    fi
    echo -e "\t[+] Virtual environment created."
fi

# Check if already in an active virtual environment
ACTIVATED=false
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$EXPECTED_ACTIVATE" ]; then
        echo "[*] Activating virtual environment: source $EXPECTED_ACTIVATE"
        source "$EXPECTED_ACTIVATE"
        ACTIVATED=true
    else
        # This case implies venv creation failed or was corrupted after check
        echo "[!] Warning: Expected activation script '$EXPECTED_ACTIVATE' not found after setup for OS type '$OS_TYPE'."
        echo "[!] Please check the virtual environment creation step for errors."
        echo "[!] Manual activation might be needed. Check .venv/bin or .venv/Scripts for activate script."
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

echo "[*] Installing dependencies..."
echo "[*] Running: $UV_PATH pip install -r requirements.txt"
"$UV_PATH" pip install -r requirements.txt
INSTALL_EXIT_CODE=$?

# Check exit code
if [ $INSTALL_EXIT_CODE -ne 0 ]; then
    echo "[!] Error: Dependency installation failed with exit code $INSTALL_EXIT_CODE."
    return 1
fi
echo -e "\t[+] Dependencies installed successfully."

echo "[*] Exporting environment variables for this session..."
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8
echo -e "\t[+] PYTHONPATH set to: $(pwd)"
echo -e "\t[+] PYTHONIOENCODING set to: $PYTHONIOENCODING"

SAMPLE_SIZE=2000
echo "[*] Generating sample data ($SAMPLE_SIZE URLs) and training the model. This may take a moment..."

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
echo "If you open a new terminal, remember to activate the venv and potentially re-export PYTHONPATH: 'uv venv && export PYTHONPATH=.'"
echo "You can also re-run 'source setup.sh' to re-export the environment variables"
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