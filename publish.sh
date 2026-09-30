#!/bin/bash
# Sonuçları yenile ve GitHub'a gönder: README tablosu + docs/index.html (GitHub Pages) + results/summary.csv
set -e
cd "$(dirname "$0")"
python3 bench.py report > /dev/null
git add README.md docs/index.html results/summary.csv results/leaderboard.md results/leaderboard.html models.tsv
git diff --cached --quiet && { echo "değişiklik yok"; exit 0; }
git commit -q -m "Update leaderboard ($(date '+%Y-%m-%d %H:%M'))"
git push -q
echo "yayınlandı: https://nsozturk.github.io/finance-llm-bench/"
