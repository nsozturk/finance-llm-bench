#!/bin/bash
# Tüm benchmark işlerini kaldığı yerden başlatır (biten modeller atlanır). Durdurmak güvenli.
cd "$(dirname "$0")"
nohup bash -c 'for i in 1 2; do python3 registry.py; python3 bench.py run; done; python3 bench.py report' >> bench_resume.log 2>&1 &
nohup python3 cli_bench.py cli-gpt-6-astra cli-gpt-5.6-sol cli-gpt-5.6-terra cli-gpt-5.6-luna >> cli_bench_resume.log 2>&1 &
echo "başladı: tail -f bench_resume.log cli_bench_resume.log"
nohup ./or_parallel.sh > /dev/null 2>&1 &   # OpenRouter ücretsiz (4 paralel) (ücretli iptal)
