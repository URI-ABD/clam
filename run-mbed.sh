#!/bin/bash
set -e

DATASET=$1
BALANCED=$2
T=$3
t=$4
B=$5
M=$6

if [ -z "$DATASET" ] || [ -z "$T" ] || [ -z "$t" ] || [ -z "$B" ] || [ -z "$M" ]; then
  echo "Usage: $0 <dataset> <balanced> <T> <t> <B> <M>"
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

cargo run -rp clam-mbed -- -i ../data/chaoda-data/datasets -o ../data/chaoda-data/mbed-results -n ${DATASET}.npy -m euclidean build ${BALANCED_FLAG} -T ${T} -t ${t} -B ${B} -M ${M}

python -m py_mbed -i ../data/chaoda-data/datasets -o ../data/chaoda-data/mbed-results -d ${DATASET}

cargo run -rp clam-mbed -- -i ../data/chaoda-data/datasets -o ../data/chaoda-data/mbed-results -n ${DATASET}.npy -m euclidean measure
