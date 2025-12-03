#!/usr/bin/env bash
set -euo pipefail

# cd into inner project root (the one that has data/, notebooks/, reports/)
cd "$(dirname "$0")/.."
echo "📍 Working directory: $(pwd)"

# 1) Clean data
python3 notebooks/02_cleaning_pipeline.py

# 2) EDA (also generates optional figures you added)
python3 notebooks/03_eda.py

# 3) QSSR models (students)
if [ -f notebooks/04_models_qssr.py ]; then
  python3 notebooks/04_models_qssr.py || true
fi

# 4) Productivity models (secondary)
if [ -f notebooks/05_secondary_productivity.py ]; then
  python3 notebooks/05_secondary_productivity.py || true
fi

echo "✅ Sprint 2 pipeline finished. See reports/ and reports/figures/."
