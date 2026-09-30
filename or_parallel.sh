#!/bin/bash
# OpenRouter modelleri: her model ayrı süreç, aynı anda 4 (kaldığı yerden devam eder). Sonra ücretli kuyruk.
cd "$(dirname "$0")"
python3 -c "import cli_bench as c;print('\n'.join(k for k in c.MODELS if k.startswith('or-')))" | xargs -P 4 -I{} sh -c 'python3 cli_bench.py {} >> cli_bench_openrouter.log 2>&1'
echo "$(date +%T) ÜCRETSİZ PARALEL BİTTİ" >> cli_bench_openrouter.log
# ücretli kuyruk kullanıcı isteğiyle iptal (2026-09-30)
