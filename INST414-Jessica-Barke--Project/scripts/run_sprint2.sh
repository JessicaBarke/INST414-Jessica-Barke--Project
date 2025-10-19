#!/usr/bin/env bash
set -euo pipefail

# --- Locate the correct project root (handles your nested folder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Candidate roots: scripts/.. (inner), and scripts/../INST414-Jessica-Barke--Project (if nested weirdly)
CANDIDATES=(
    "$SCRIPT_DIR/.."
    "$SCRIPT_DIR/../INST414-Jessica-Barke--Project"
)

PROJECT_ROOT=""
for C in "${CANDIDATES[@]}"; do
    if [[ -d "$C/notebooks" || -d "$C/cards" || -d "$C/data" ]]; then
        PROJECT_ROOT="$C"
        break
        fi
done

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "❌ Could not find project root. Make sure this script lives in scripts/ under your project."
    exit 1
fi

cd "$PROJECT_ROOT"
echo "📍 Working directory set to: $(pwd)"

# --- Ensure folders exist ---
mkdir -p data/raw data/clean reports/figures cards/addiction_card cards/productivity_card notebooks

# --- (Optional) ensure deps are installed ---
# pip install -r requirements.txt || true

# --- Step 0: pull data + cards ---
if [[ ! -x scripts/get_data.sh ]]; then
    echo "❌ scripts/get_data.sh not found or not executable."
    echo "   Create it and chmod +x scripts/get_data.sh, then rerun."
    exit 1
fi
./scripts/get_data.sh

# --- Step 1: Acquisition & inventories ---
if [[ -f notebooks/01_data_acquisition.py ]]; then
    python3 notebooks/01_data_acquisition.py
else
    echo "⚠️ missing notebooks/01_data_acquisition.py (continuing)"
fi

if [[ -f notebooks/make_variable_inventory_qssr.py ]]; then
    python3 notebooks/make_variable_inventory_qssr.py
else
    echo "⚠️ missing notebooks/make_variable_inventory_qssr.py (continuing)"
fi

# --- Step 2: Cleaning ---
if [[ -f notebooks/02_cleaning_pipeline.py ]]; then
    python3 notebooks/02_cleaning_pipeline.py
else
echo "⚠️ missing notebooks/02_cleaning_pipeline.py (continuing)"
fi

# --- Step 3: EDA ---
if [[ -f notebooks/03_eda.py ]]; then
    python3 notebooks/03_eda.py
else
    echo "⚠️ missing notebooks/03_eda.py (continuing)"
fi

# --- Step 4: QSSR models ---
if [[ -f notebooks/04_models_qssr.py ]]; then
    python3 notebooks/04_models_qssr.py
else
    echo "⚠️ missing notebooks/04_models_qssr.py (continuing)"
fi

echo "✅ Sprint 2 complete. Outputs in ./reports/"
echo "   - tables: CSV/MD/TXT in reports/"
echo "   - figures: PNGs in reports/figures/"
