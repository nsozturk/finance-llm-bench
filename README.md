# finance-llm-bench

A judge-free benchmark of **finance-tuned LLMs, open general models and frontier models** on seven public finance tasks — run on a single 64 GB Apple M1 Max, with frontier and hosted models queried through their CLIs.

**Interactive leaderboard:** https://nsozturk.github.io/finance-llm-bench/

- 835 questions per model across sentiment and numerical-reasoning tasks
- Local models: 4-bit GGUF (llama.cpp) or MLX, served one at a time on the Mac
- Frontier models: Gemini, GPT (codex), GLM, Muse via their CLIs; free and cheap models via OpenRouter
- Scoring is rule-based (no LLM judge); every run is resumable and journaled per question

## Leaderboard

Overall = mean of the sentiment block and the numeric block (0–100, higher is better). `tok/s` is single-stream generation speed on the M1 Max (local models only).

<!-- LEADERBOARD:START -->
_Updated 2026-09-30 23:18 · 58 models complete · 835 questions per model_

| # | Model | Group | Quant | Overall | Sentiment | Numeric | FPB | FiQA-SA | TFNS | FinQA | ConvFinQA | TAT-QA | FinReason | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cli-gemini-3.8-flash-high | frontier (CLI) | API | **87.3** | 82.9 | 91.7 | 83.0 | 84.3 | 81.5 | 92.0 | 92.0 | 90.8 | 92.0 | – |
| 2 | cli-gpt-6-astra | frontier (CLI) | API | **86.1** | 82.7 | 89.5 | 85.5 | 82.1 | 80.5 | 90.0 | 94.0 | 86.2 | 88.0 | – |
| 3 | or-qwen3.8-27b | OpenRouter free | API | **83.7** | 79.1 | 88.2 | 81.0 | 84.7 | 71.5 | 92.0 | 92.0 | 83.0 | 86.0 | – |
| 4 | cli-gemini-3.1-pro-high | frontier (CLI) | API | **83.0** | 77.0 | 89.0 | 78.5 | 85.1 | 67.5 | 90.0 | 88.0 | 88.2 | 90.0 | – |
| 5 | cli-glm-5.3 | frontier (CLI) | API | **82.8** | 76.0 | 89.6 | 75.5 | 82.6 | 70.0 | 92.0 | 88.0 | 86.4 | 92.0 | – |
| 6 | cli-muse-spark-1.3 | frontier (CLI) | API | **82.5** | 76.1 | 89.0 | 78.0 | 81.7 | 68.5 | 88.0 | 92.0 | 88.1 | 88.0 | – |
| 7 | cli-gpt-5.6-luna | frontier (CLI) | API | **81.6** | 74.2 | 89.0 | 72.0 | 85.5 | 65.0 | 92.0 | 90.0 | 84.2 | 90.0 | – |
| 8 | or-dots-3-note-preview | OpenRouter free | API | **81.2** | 75.2 | 87.3 | 78.5 | 83.0 | 64.2 | 84.0 | 90.0 | 87.3 | 88.0 | – |
| 9 | cli-gpt-5.6-sol | frontier (CLI) | API | **81.2** | 73.5 | 89.0 | 76.5 | 82.6 | 61.5 | 92.0 | 90.0 | 84.0 | 90.0 | – |
| 10 | or-inkling | OpenRouter free | API | **81.1** | 75.7 | 86.5 | 79.5 | 80.0 | 67.5 | 86.0 | 86.0 | 84.0 | 90.0 | – |
| 11 | orp-ling-3.0-flash | OpenRouter paid | API | **81.0** | 74.9 | 87.0 | 74.0 | 83.8 | 67.0 | 82.0 | 88.0 | 84.5 | 93.3 | – |
| 12 | or-ling-3.0-flash-sante | OpenRouter free | API | **80.8** | 75.1 | 86.5 | 76.5 | 84.3 | 64.5 | 84.0 | 88.0 | 86.6 | 87.5 | – |
| 13 | cli-gpt-5.6-terra | frontier (CLI) | API | **80.0** | 73.5 | 86.5 | 75.0 | 83.0 | 62.5 | 94.0 | 88.0 | 79.8 | 84.0 | – |
| 14 | or-nemotron-3-ultra-550b-a55b | OpenRouter free | API | **79.5** | 72.9 | 86.0 | 70.0 | 82.1 | 66.5 | 88.0 | 86.0 | 82.2 | 88.0 | – |
| 15 | Qwen3.8-27B-Heretic-Abliterated-Uncensored | uncensored | Q4_K_M | **79.3** | 78.5 | 80.2 | 82.0 | 83.4 | 70.0 | 84.0 | 82.0 | 86.7 | 68.0 | 14 |
| 16 | google_gemma-4-31B-it | open general | Q4_K_M | **78.4** | 70.5 | 86.3 | 79.0 | 73.6 | 59.0 | 82.0 | 86.0 | 89.3 | 88.0 | 11 |
| 17 | Qwen_Qwen3.6-35B-A3B | open general | Q4_K_M | **78.1** | 76.4 | 79.7 | 82.0 | 79.1 | 68.0 | 76.0 | 86.0 | 76.6 | 80.0 | 24 |
| 18 | Huihui-Qwen3.8-27B-abliterated-GGUF | uncensored | IQ3_S | **77.8** | 75.9 | 79.8 | 80.5 | 83.8 | 63.5 | 84.0 | 84.0 | 87.0 | 64.0 | 15 |
| 19 | Qwen3.5-35B-A3B-4bit | open general | 4bit | **76.9** | 75.5 | 78.3 | 80.0 | 80.0 | 66.5 | 74.0 | 86.0 | 75.4 | 78.0 | – |
| 20 | Qwen3.8-27B-Fable-Distill-Heretic-ara | uncensored | Q4_K_S | **76.8** | 72.9 | 80.6 | 75.5 | 81.7 | 61.5 | 90.0 | 86.0 | 84.3 | 62.0 | 13 |
| 21 | or-nemotron-3-super-120b-a12b | OpenRouter free | API | **76.8** | 70.1 | 83.6 | 69.0 | 84.7 | 56.5 | 88.0 | 84.0 | 80.3 | 82.0 | – |
| 22 | or-laguna-s-2.1 | OpenRouter free | API | **76.7** | 74.5 | 78.8 | 72.0 | 86.0 | 65.5 | 84.0 | 72.0 | 79.2 | 80.0 | – |
| 23 | gemma-4-26B-A4B-it | open general | Q4_K_M | **76.7** | 73.2 | 80.2 | 78.5 | 79.1 | 62.0 | 76.0 | 80.0 | 84.8 | 80.0 | 22 |
| 24 | FinSenti-Qwen3.5-9B | finance-tuned | Q4_K_M | **75.2** | 77.2 | 73.2 | 79.0 | 81.7 | 71.0 | 86.0 | 78.0 | 72.7 | 56.0 | 42 |
| 25 | Mihenk-LLM-v2-35B-A3B-Turkish-Financial-Model | finance-tuned | Q4_K_M | **74.9** | 75.8 | 74.1 | 82.0 | 78.3 | 67.0 | 74.0 | 84.0 | 72.4 | 66.0 | 24 |
| 26 | gpt-oss-20b-finance | finance-tuned | Q4_K_M | **74.6** | 69.9 | 79.2 | 80.5 | 78.3 | 51.0 | 76.0 | 90.0 | 82.6 | 68.0 | 18 |
| 27 | orp-gpt-oss-20b | OpenRouter paid | API | **74.6** | 73.2 | 76.0 | 74.0 | 82.6 | 63.0 | 80.0 | 70.0 | 71.9 | 82.0 | – |
| 28 | google_gemma-4-E4B-it | open general | Q4_K_M | **72.7** | 70.6 | 74.8 | 74.0 | 71.9 | 66.0 | 80.0 | 82.0 | 83.4 | 54.0 | 55 |
| 29 | gpt-oss-20b | open general | MXFP4 | **72.5** | 69.2 | 75.8 | 82.0 | 74.0 | 51.5 | 74.0 | 88.0 | 77.1 | 64.0 | 23 |
| 30 | or-inkling-small | OpenRouter free | API | **71.2** | 64.5 | 77.8 | 63.5 | 80.4 | 49.5 | 86.0 | 80.0 | 85.3 | 60.0 | – |
| 31 | Qwen3.5-4B-Financial-SQL | finance-tuned | Q4_K_M | **70.7** | 68.2 | 73.2 | 73.0 | 77.0 | 54.5 | 80.0 | 82.0 | 72.7 | 58.0 | 25 |
| 32 | ODA-Fin-SFT-8B | finance-tuned | Q4_K_M | **70.7** | 73.5 | 67.8 | 77.5 | 77.4 | 65.5 | 72.0 | 80.0 | 73.0 | 46.0 | 20 |
| 33 | SwayAlgo-Finance-gemma-4-E4B-it-1 | finance-tuned | Q4_K_M | **69.7** | 68.9 | 70.4 | 73.5 | 78.7 | 54.5 | 74.0 | 70.0 | 77.5 | 60.0 | 13 |
| 34 | google_gemma-4-E2B-it | open general | Q4_K_M | **67.5** | 69.5 | 65.5 | 75.5 | 69.4 | 63.5 | 68.0 | 76.0 | 64.0 | 54.0 | 13 |
| 35 | Qwen3-30B-A3B-Finance | finance-tuned | Q4_K_M | **67.0** | 65.0 | 68.9 | 67.0 | 80.4 | 47.5 | 76.0 | 74.0 | 67.5 | 58.0 | 25 |
| 36 | Fin-R1 | finance-tuned | Q4_K_M | **66.3** | 67.4 | 65.3 | 71.5 | 73.2 | 57.5 | 72.0 | 82.0 | 69.3 | 38.0 | 46 |
| 37 | GLM-4.7-Flash-MLX-6bit | open general | 6bit | **65.0** | 61.9 | 68.2 | 69.0 | 84.3 | 32.5 | 72.0 | 82.0 | 68.7 | 50.0 | – |
| 38 | BIST-Financial-Qwen-7B | finance-tuned | Q4_K_M | **64.8** | 68.4 | 61.1 | 77.5 | 62.6 | 65.0 | 80.0 | 62.0 | 66.3 | 36.0 | 17 |
| 39 | finance-gemma4-e2b | finance-tuned | Q4_K_M | **63.2** | 67.4 | 59.1 | 75.5 | 72.3 | 54.5 | 66.0 | 68.0 | 70.3 | 32.0 | 40 |
| 40 | BIST-LLM-8B-Turkish-Finance | finance-tuned | Q4_K_M | **61.9** | 72.0 | 51.8 | 72.5 | 68.5 | 75.0 | 70.0 | 72.0 | 51.3 | 14.0 | 12 |
| 41 | Llama-3.1-8B-Instruct | open general | Q4_K_M | **60.9** | 71.5 | 50.3 | 69.5 | 73.6 | 71.5 | 72.0 | 66.0 | 45.3 | 18.0 | 11 |
| 42 | TraceAlchemy-Gemma-4-E4B-Finance-IT | finance-tuned | Q4_K_M | **60.5** | 67.5 | 53.5 | 68.0 | 79.6 | 55.0 | 58.0 | 48.0 | 70.1 | 38.0 | 13 |
| 43 | Blum-Finance-4B | finance-tuned | Q4_K_M | **59.5** | 60.6 | 58.4 | 66.5 | 76.2 | 39.0 | 68.0 | 66.0 | 61.5 | 38.0 | 28 |
| 44 | Phinance-Phi-4-mini-instruct-finance-v0.4-with-reasoning | finance-tuned | Q4_K_M | **59.3** | 73.6 | 45.1 | 75.0 | 71.9 | 74.0 | 58.0 | 52.0 | 62.3 | 8.0 | 72 |
| 45 | LFM2.5-2.6B-Finance | finance-tuned | Q4_K_M | **58.1** | 63.7 | 52.5 | 75.0 | 64.7 | 51.5 | 64.0 | 60.0 | 58.2 | 28.0 | 40 |
| 46 | or-north-mini-code | OpenRouter free | API | **54.2** | 69.0 | 39.4 | 72.0 | 80.4 | 54.5 | 52.0 | 28.0 | 67.7 | 10.0 | – |
| 47 | gemma-4-E4B-medical-legal-finance-qa | finance-tuned | Q4_K_M | **54.2** | 57.2 | 51.2 | 58.0 | 78.7 | 35.0 | 60.0 | 56.0 | 64.9 | 24.0 | 13 |
| 48 | or-laguna-xs-2.1 | OpenRouter free | API | **53.0** | 66.0 | 39.9 | 65.0 | 82.6 | 50.5 | 42.0 | 40.0 | 67.7 | 10.0 | – |
| 49 | Llama-Open-Finance-8B-Q4_K_M | finance-tuned | Q4_K_M | **49.8** | 53.7 | 45.8 | 43.5 | 70.6 | 47.0 | 60.0 | 46.0 | 63.1 | 14.0 | 29 |
| 50 | finanalyzer-indian-bank-statements | finance-tuned | Q4_K_M | **39.5** | 58.4 | 20.6 | 66.5 | 72.8 | 36.0 | 8.0 | 24.0 | 48.6 | 2.0 | 47 |
| 51 | bitcoin-news-1.7b | finance-tuned | Q4_K_S | **39.5** | 55.7 | 23.3 | 70.5 | 67.2 | 29.5 | 18.0 | 20.0 | 51.2 | 4.0 | 59 |
| 52 | FinanceGemma-E4B | finance-tuned | Q4_K_M | **38.6** | 68.9 | 8.2 | 64.0 | 67.7 | 75.0 | 4.0 | 14.0 | 12.8 | 2.0 | 49 |
| 53 | tanpo-finance | finance-tuned | Q4_K_M | **33.2** | 46.7 | 19.8 | 47.0 | 65.1 | 28.0 | 16.0 | 20.0 | 37.0 | 6.0 | 81 |
| 54 | finma-7b-full | finance-tuned | Q4_K_M | **33.0** | 47.9 | 18.1 | 41.0 | 78.3 | 24.5 | 22.0 | 28.0 | 22.5 | 0.0 | 21 |
| 55 | finance-chat | finance-tuned | Q4_K_M | **32.6** | 45.5 | 19.8 | 40.0 | 37.9 | 58.5 | 14.0 | 18.0 | 43.0 | 4.0 | 19 |
| 56 | finance-LLM-13B | finance-tuned | Q4_K_M | **31.3** | 45.8 | 16.9 | 50.5 | 20.4 | 66.5 | 12.0 | 6.0 | 47.8 | 2.0 | 27 |
| 57 | finance-Llama3-8B | finance-tuned | Q4_K_M | **24.2** | 45.7 | 2.7 | 44.5 | 47.2 | 45.5 | 2.0 | 2.0 | 2.8 | 4.0 | 18 |
| 58 | finance-LLM | finance-tuned | Q4_K_M | **0.9** | 0.0 | 1.9 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 5.5 | 0.0 | 49 |
<!-- LEADERBOARD:END -->

## Tasks

| Task | Block | Questions | What it measures | Metric |
|---|---|---|---|---|
| [FPB](https://huggingface.co/datasets/takala/financial_phrasebank) | Sentiment | 200 | Financial PhraseBank: sentiment of news sentences | accuracy |
| [FiQA-SA](https://huggingface.co/datasets/AdaptLLM/finance-tasks) | Sentiment | 235 | Target-aware sentiment of microblogs / headlines | accuracy |
| [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Sentiment | 200 | Twitter financial news: bullish / bearish / neutral | accuracy |
| [FinQA](https://huggingface.co/datasets/TheFinAI/FINQA_test_test) | Numeric | 50 | Numerical questions over report tables + text | numeric match, 1 % tolerance |
| [ConvFinQA](https://huggingface.co/datasets/TheFinAI/flare-convfinqa) | Numeric | 50 | Multi-turn chained numerical questions | numeric match, 1 % tolerance |
| [TAT-QA](https://huggingface.co/datasets/TheFinAI/flare-tatqa) | Numeric | 50 | Hybrid table + text QA (number or span) | numeric match / token F1 |
| [FinanceReasoning (hard)](https://github.com/BUPT-Reasoning-Lab/FinanceReasoning) | Numeric | 50 | Multi-step financial calculations | numeric match, 1 % tolerance |

Each task is a fixed random sample (seed 1234) of the public test split. Sentiment prompts are zero-shot.

## Method

- **No judge model.** Numeric answers are extracted from the final `Answer:` line and accepted within 1 % relative error; a percentage and its decimal form (14.5 ↔ 0.145) count as the same answer. Text answers in TAT-QA use token F1.
- **Deterministic settings.** Temperature 0, seed 1234. Thinking is switched off where the chat template allows it; reasoning-only models get a 1024-token thinking budget.
- **Local models** run on `llama-server` (GGUF, Metal) or `mlx_lm server` (MLX), four parallel slots, one model at a time. Models below 10 tok/s single-stream are not benchmarked.
- **Frontier and hosted models** are queried through their CLI agents (codex, opencode, agy, muse) in batches — 80 sentiment or 10 numeric questions per call — with tool use forbidden in the prompt. OpenRouter models without tool-use support fall back to the plain chat API.

## Known limitations

- Frontier / OpenRouter rows are **not run under identical conditions** to the local rows: they see questions in batches, and tool use is forbidden only by prompt.
- Task sizes are small (50–235 questions). Differences of 2–3 points are within noise.
- **Base (non-chat) models** such as AdaptLLM `finance-LLM` or `deepmoney-34b-200k-base` do not follow the chat format used here and score near zero; their papers evaluate them with few-shot completion prompts.
- Local models are 4-bit quantized; the comparison of `or-qwen3.8-27b` (full model via API) with the local Qwen3.8 variants shows the combined cost of quantization and abliteration.

## Reproduce

```bash
python3 registry.py          # build models.tsv from local model folders (fastest first)
python3 bench.py run         # local models; resumable
python3 cli_bench.py         # frontier / OpenRouter models via CLIs; resumable
python3 bench.py report      # results/summary.csv, docs/index.html and the table above
```

Datasets are **not** included in this repository. Download them from the links in the Tasks table into `data/` (see `load_tasks()` in `bench.py` for the expected paths). Several of them are licensed for non-commercial use only (FPB CC-BY-NC-SA-3.0; AdaptLLM finance-tasks; FinanceReasoning CC-BY-4.0) — check each license before use.

Per-question outputs (`results/raw/`) are kept local because they contain gold answers from the datasets.

## License

Code: MIT. Benchmark data belongs to the respective dataset authors under their own licenses.
