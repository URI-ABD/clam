#!/bin/bash
set -e

DATASET=$1  # The dataset name (e.g., "cardio")
BALANCED=$2 # "true" or "false"
T=$3        # Target stability value
t=$4        # Time step
B=$5        # Damping constant
M=$6        # Maximum number of steps
k=$7        # Spring constant
P=$8        # Patience

if [ -z "$DATASET" ] || [ -z "$T" ] || [ -z "$t" ] || [ -z "$B" ] || [ -z "$M" ] || [ -z "$k" ] || [ -z "$P" ]; then
  echo "Usage: $0 <dataset> <balanced> <T> <t> <B> <M> <k> <P>"
  exit 1
fi

BALANCED_FLAG=""
if [ "$BALANCED" == "true" ]; then
  BALANCED_FLAG="-b"
fi

TREE_FILE="../data/chaoda-data/mbed-results/${DATASET}/${DATASET}-tree.bin"
if [ -f "$TREE_FILE" ]; then
  rm "$TREE_FILE"
fi

cargo run -rp clam-mbed -- \
  -i ../data/chaoda-data/datasets \
  -o ../data/chaoda-data/mbed-results \
  -n ${DATASET}.npy \
  -m euclidean build ${BALANCED_FLAG} \
  -T ${T} -t ${t} -B ${B} -M ${M} -k ${k} -p ${P}

uv run python -m py_mbed \
  -i ../data/chaoda-data/datasets \
  -o ../data/chaoda-data/mbed-results \
  -d ${DATASET}

cargo run -rp clam-mbed -- \
  -i ../data/chaoda-data/datasets \
  -o ../data/chaoda-data/mbed-results \
  -n ${DATASET}.npy \
  -m euclidean measure
