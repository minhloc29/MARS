#!/usr/bin/env bash
set -e

# Script to automatically download and build LKH-3 for Linux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading LKH-3 source code..."
curl -sO http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz

echo "Extracting..."
tar -xzf LKH-3.0.14.tgz
cd LKH-3.0.14

echo "Compiling LKH-3 with make..."
make -j$(nproc || echo 2)

echo "Copying binary to baselines/lkh/LKH..."
cp LKH "$SCRIPT_DIR/LKH"
chmod +x "$SCRIPT_DIR/LKH"

cd "$SCRIPT_DIR"
rm -rf LKH-3.0.14 LKH-3.0.14.tgz

echo "[OK] LKH compiled successfully at: $SCRIPT_DIR/LKH"
"$SCRIPT_DIR/LKH" || true
