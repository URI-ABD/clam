#!/bin/sh
set -e

echo "Configuring git hooks."
git config core.hooksPath .hooks
chmod +x .hooks/*
