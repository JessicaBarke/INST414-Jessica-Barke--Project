#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw cards/addiction_card cards/productivity_card

# Download the two datasets
kaggle datasets download -d adilshamim8/social-media-addiction-vs-relationships -p data/raw --unzip
kaggle datasets download -d mahdimashayekhi/social-media-vs-productivity -p data/raw --unzip

# Normalize the primary filename (Kaggle unzips with spaces)
if [ -f "data/raw/Students Social Media Addiction.csv" ]; then
  mv "data/raw/Students Social Media Addiction.csv" data/raw/social_media_addiction_vs_relationships.csv
fi

# Download dataset metadata ("data cards")
kaggle datasets metadata adilshamim8/social-media-addiction-vs-relationships -p cards/addiction_card
kaggle datasets metadata mahdimashayekhi/social-media-vs-productivity       -p cards/productivity_card

# Convert metadata JSONs to one-row CSVs
python3 cards/card2csv.py

echo "✅ Data and cards ready:"
ls -la data/raw
