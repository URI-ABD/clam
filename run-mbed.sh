#!/bin/bash
set -e

# Parameters for all commands
DATASET=$1  # The dataset name (e.g., "cardio")
METRIC=$2   # The distance metric to use (e.g., "euclidean")
SEED=$3     # Random seed for reproducibility
LOG_NAME=$4 # Base name for log files

# Parameters specific to MBED Build
BALANCED=$5 # "true" or "false"
BETA=$6     # Damping constant
k=$7        # Starting Spring constant
DK=$8        # Factor by which to decrease the spring constant
DT=$9        # Time step size for each iteration
PATIENCE=${10} # Number of iterations to wait for improvement before stopping
TARGET=${11}  # Target stability value
MAX_STEPS=${12} # Maximum number of iterations

# Parameters specific to MBED Evaluate
MEASURE=${13}  # The quality measure to use (e.g., "fnn")
EXHAUSTIVE=${14} # "true" or "false"

# Check if all required parameters are provided
if [ -z "$DATASET" ] || [ -z "$METRIC" ] || [ -z "$SEED" ] || [ -z "$LOG_NAME" ] || [ -z "$BALANCED" ] || [ -z "$BETA" ] || [ -z "$k" ] || [ -z "$DK" ] || [ -z "$DT" ] || [ -z "$PATIENCE" ] || [ -z "$TARGET" ] || [ -z "$MAX_STEPS" ] || [ -z "$MEASURE" ] || [ -z "$EXHAUSTIVE" ]; then
  echo "Usage: $0 <dataset> <metric> <seed> <log_name> <balanced> <beta> <k> <DK> <DT> <PATIENCE> <TARGET> <MAX_STEPS> <MEASURE> <EXHAUSTIVE>"
  exit 1
fi

# Create input and output paths
INP_PATH="../data/chaoda-data/datasets/${DATASET}.npy"
if [ ! -f "$INP_PATH" ]; then
  echo "Input dataset file not found: $INP_PATH"
  exit 1
fi

OUT_ROOT="../data/chaoda-data/mbed-results"
mkdir -p "$OUT_ROOT"

OUT_DIR="${OUT_ROOT}/${DATASET}"
mkdir -p "$OUT_DIR"

# Determine flags based on boolean parameters
BALANCED_FLAG=""
if [ "$BALANCED" == "true" ] || [ "$BALANCED" == "1" ]; then
  BALANCED_FLAG="-b"
fi

EXHAUSTIVE_FLAG=""
if [ "$EXHAUSTIVE" == "true" ] || [ "$EXHAUSTIVE" == "1" ]; then
  EXHAUSTIVE_FLAG="-e"
fi

# # Run the MBED algorithm to create the dimension reduction
# cargo run -rp shell -- \
#   -i "$INP_PATH" \
#   -m "$METRIC" \
#   -s "$SEED" \
#   -l "$LOG_NAME" \
#   mbed build \
#   -o "$OUT_DIR" \
#   $BALANCED_FLAG \
#   -B "$BETA" \
#   -k "$k" \
#   -K "$DK" \
#   -t "$DT" \
#   -p "$PATIENCE" \
#   -T "$TARGET" \
#   -M "$MAX_STEPS"

# Run the Python visualization and comparison against UMAP
uv run py-mbed \
  -i ../data/chaoda-data/datasets \
  -o ../data/chaoda-data/mbed-results \
  -d ${DATASET} \
  -m ${METRIC}

# Evaluate the quality of the dimension reduction
cargo run -rp shell -- \
  -i "$INP_PATH" \
  -m "$METRIC" \
  -s "$SEED" \
  -l "$LOG_NAME" \
  mbed evaluate \
  -o "$OUT_DIR" \
  -M "$MEASURE" \
  $EXHAUSTIVE_FLAG
