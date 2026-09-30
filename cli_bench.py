#!/usr/bin/env python3
"""Aynı Tier-1 görevleri frontier modellerle, abonelik CLI'ları üzerinden koşar (codex / opencode GLM / agy Gemini / muse).
  python3 cli_bench.py            kaldığı yerden devam eder (results/raw/cli-*/<görev>.jsonl)
Sorular paketler halinde gider (CLI ajanları her çağrıda ~50k token başlangıç yükü taşır). Skorlama bench.py ile aynı.
Kota kuralı: bir sağlayıcı arka arkaya 3 paket başarısız olursa o sağlayıcı DURUR ve 'BLOCKED' yazılır.
"""
import os, re, sys, json, time, subprocess, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
import bench as B

SCRATCH = f"{B.ROOT}/stage/cli-scratch"          # boş çalışma klasörü (ajanlar dosya görmesin)
AGY = "/Users/ns0bj/.local/bin/agy"
BATCH = {"sent": 80, "num": 10, "tat": 10}
MODELS = {  # id: (sağlayıcı, model, effort)
    "cli-glm-5.3":               ("opencode", "zai-coding-plan/glm-5.3", None),
    "cli-muse-spark-1.3":        ("muse", "muse-spark-1.3-contributor", "high"),
    "cli-gemini-3.8-flash-high": ("agy", "Gemini 3.8 Flash (High)", None),
    "cli-gemini-3.1-pro-high":   ("agy", "Gemini 3.1 Pro (High)", None),
    "cli-gpt-6-astra":           ("codex", "gpt-6-astra", "high"),
    "cli-gpt-5.6-sol":           ("codex", "gpt-5.6-sol", "high"),
    "cli-gpt-5.6-terra":         ("codex", "gpt-5.6-terra", "high"),
    "cli-gpt-5.6-luna":          ("codex", "gpt-5.6-luna", "max"),     # Luna sadece max
}
# OpenRouter ücretsiz modeller (opencode üzerinden); içerik güvenliği / müzik / router girdileri hariç
for _m in ["qwen/qwen3.8-27b", "google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it", "nvidia/nemotron-3-ultra-550b-a55b",
           "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3.5-lightning", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
           "thinkingmachines/inkling", "thinkingmachines/inkling-small", "inclusionai/ling-3.0-flash-sante",
           "dots-studio/dots-3-note-preview", "poolside/laguna-s-2.1", "poolside/laguna-xs-2.1", "cohere/north-mini-code",
           "liquid/lfm-2.5-2.6b"]:
    MODELS["or-" + _m.split("/")[1]] = ("openrouter", f"openrouter/{_m}:free", None)
HEAD = """You are being evaluated on a financial NLP benchmark.
Do NOT use any tools: do not read or write files, do not run code or shell commands, do not search the web. Answer every item only from its own text.
Give the final answer for each item:
- sentiment items: exactly one of Positive, Negative, Neutral
- numeric items: the final value as a plain number without units, currency or % sign (a percentage as its percent number, e.g. 14.5), or yes/no, or a short text span when the question asks for text
Return ONLY a JSON array with one object per item, in a ```json code block: [{"id": "<id>", "answer": "<answer>"}]

ITEMS:
"""

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

def call(provider, model, effort, prompt, tag):
    pf = f"{SCRATCH}/{tag}.prompt"; open(pf, "w").write(prompt)
    out = f"{SCRATCH}/{tag}.out"
    if provider == "codex":
        cmd = ["codex", "exec", "-C", SCRATCH, "-s", "read-only", "--skip-git-repo-check", "-m", model,
               "-c", f'model_reasoning_effort="{effort}"', "-o", out, prompt]
    elif provider in ("opencode", "openrouter", "openrouter-paid"):
        cmd = ["opencode", "run", "--pure", "--auto", "--title", f"bench {tag}", "-m", model, prompt]   # --title: otomatik başlık üretimi (ek ücretli çağrı) olmasın
    elif provider == "agy":
        cmd = [AGY, "--dangerously-skip-permissions", "--disable-slash-commands", "--print-timeout", "20m",
               "--add-dir", SCRATCH, "--model", model, "--prompt", prompt]
    else:
        cmd = ["muse", "exec", "--model", model, "--reasoning-effort", effort, "--workspace", SCRATCH, "--prompt-file", pf]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=SCRATCH, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
    text = open(out).read() if provider == "codex" and os.path.exists(out) else r.stdout
    if provider.startswith("openrouter") and "support tool use" in text + r.stderr:      # opencode araç ister; araçsız modelde doğrudan API
        body = {"model": model.removeprefix("openrouter/"), "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + or_key()})
        try:
            j = json.load(urllib.request.urlopen(req, timeout=900))
            return 0, j["choices"][0]["message"].get("content") or "", "", time.time() - t0
        except Exception as e:
            return 1, "", f"api: {e}", time.time() - t0
    return r.returncode, text, r.stderr[-800:], time.time() - t0

def parse(text):
    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S) or re.findall(r"(\[\s*\{.*\}\s*\])", text, re.S)
    for b in reversed(blocks):
        try:
            return {str(o["id"]): str(o.get("answer", "")) for o in json.loads(b) if isinstance(o, dict) and "id" in o}
        except Exception: continue
    return None

PAID_CAP_USD = 5.0          # ücretli OpenRouter aşaması için toplam harcama sınırı (hesabın gerçek kullanımına göre)
PAID_MAX_EST_USD = 0.5      # tahmini maliyeti bundan yüksek tek model atlanır
PAID_SKIP = re.compile(r":batch|guard|safety|embed|rerank|moderation|schematron|mythomax|lunaris|hy-mt|-vl$|^~|lyria|image|audio|ui-tars|voxtral|cydonia|dolphin|muse-spark", re.I)

def or_key():
    return (json.load(open(os.path.expanduser("~/.local/share/opencode/auth.json"))).get("openrouter") or {}).get("key")

def or_usage():
    r = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + or_key()})
    return float(json.load(urllib.request.urlopen(r, timeout=20))["data"]["usage"])

def paid_queue():
    """OpenRouter ücretli metin modelleri, ucuzdan pahalıya; opencode'da olmayanlar ve ücretsiz eşi test edilenler atlanır."""
    models = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30))["data"]
    have = set(subprocess.run(["opencode", "models"], stdin=subprocess.DEVNULL, capture_output=True, text=True).stdout.split())
    free_bases = {v[1].removeprefix("openrouter/").removesuffix(":free") for k, v in MODELS.items()
                  if v[0] == "openrouter" and os.path.exists(f"{B.RAW}/{k}/DONE")}   # ücretsiz testi bitmemişse ücretliyi koş
    out = []
    for m in models:
        pr = m["pricing"]; pi, po = float(pr.get("prompt", 0)) * 1e6, float(pr.get("completion", 0)) * 1e6
        if pi <= 0 and po <= 0 or not (m.get("architecture") or {}).get("modality", "").endswith("->text"): continue
        if PAID_SKIP.search(m["id"]) or m["id"] in free_bases or f"openrouter/{m['id']}" not in have: continue
        est = pi * 1.0 + po * 0.3                      # ~1M giriş + ~0.3M çıkış token / model
        if est > PAID_MAX_EST_USD: continue
        out.append((est, m["id"]))
    return [i for _, i in sorted(out)]

class Blocked(Exception): pass

def run_model(mid, tasks, fails):
    provider, model, effort = MODELS[mid]
    d = f"{B.RAW}/{mid}"; os.makedirs(d, exist_ok=True)
    meta = f"{d}/meta.json"
    if not os.path.exists(meta):
        json.dump({"id": mid, "group": {"openrouter": "openrouter-free", "openrouter-paid": "openrouter-paid"}.get(provider, "frontier-cli"), "provider": provider, "model": model, "effort": effort,
                   "mode": "batched via CLI agent, tools forbidden by prompt", "batch": BATCH}, open(meta, "w"), indent=1)
    for key, items in tasks.items():
        jp = f"{d}/{key}.jsonl"
        done = set()
        if os.path.exists(jp):
            for l in open(jp):
                try: done.add(json.loads(l)["id"])
                except Exception: pass
        todo = [x for x in items if x["id"] not in done]
        if not todo: continue
        n = BATCH[todo[0]["kind"]]
        log(f"{mid} {key}: {len(done)} tamam, {len(todo)} kaldı ({-(-len(todo)//n)} paket)")
        for i in range(0, len(todo), n):
            chunk = todo[i:i + n]
            q = "\n".join(json.dumps({"id": x["id"], "question": x["prompt"].replace(B.NUM_INSTR, "").replace(B.SENT_INSTR, "")},
                                     ensure_ascii=False) for x in chunk)
            for attempt in range(5):
                try:
                    rc, text, err, dt = call(provider, model, effort, HEAD + q, f"{mid}_{key}_{i}")
                except subprocess.TimeoutExpired:
                    rc, text, err, dt = -1, "", "timeout", 1800
                ans = parse(text) if rc == 0 else None
                if ans or not re.search(r"rate.?limit|429|temporarily", text + err, re.I): break
                log(f"  {mid} {key} paket {i // n}: geçici hız sınırı, 90 sn sonra tekrar ({attempt + 1}/4)")
                time.sleep(90)
            err = err or re.sub(r"\x1b\[[0-9;]*m", "", text)[-300:]      # opencode hatayı stdout'a yazar
            if not ans:
                fails[provider] += 1
                log(f"  {mid} {key} paket {i // n}: CEVAP YOK (rc={rc}) {err.strip()[-300:]}")
                if fails[provider] >= 3: raise Blocked(f"{provider}: 3 paket üst üste başarısız — son hata: {err.strip()[-300:]}")
                continue
            fails[provider] = 0
            with open(jp, "a") as jf:
                for x in chunk:
                    if x["id"] not in ans: continue                   # eksik kalan, sonraki çalıştırmada tekrar sorulur
                    pred, sc = B.score(x["kind"], "Answer: " + ans[x["id"]], x["gold"])
                    jf.write(json.dumps({"id": x["id"], "gold": x["gold"], "pred": pred, "score": sc, "contended": 0,
                                         "text": ans[x["id"]], "finish": "stop", "tok_in": None, "tok_out": None,
                                         "lat": round(dt / len(chunk), 2), "tps": None}, ensure_ascii=False) + "\n")
    open(f"{d}/DONE", "w").write(time.strftime("%F %T"))
    log(f"tamam: {mid}")

def provider_worker(provider, tasks, fails, want):
    for mid, (p, _, _) in MODELS.items():
        if p != provider or (want and mid not in want): continue
        try: run_model(mid, tasks, fails)
        except Blocked as e:
            log(f"BLOCKED {e}"); return
        except Exception as e:
            log(f"HATA {mid}: {e}")

def run_paid(tasks):
    q = paid_queue(); start = or_usage()
    log(f"ücretli kuyruk: {len(q)} model, başlangıç kullanımı ${start:.3f}, sınır ${PAID_CAP_USD}")
    fails = {"openrouter-paid": 0}
    for mid_or in q:
        spent = or_usage() - start
        if spent >= PAID_CAP_USD: log(f"BÜTÇE DOLDU: ${spent:.2f} harcandı, durdu"); break
        mid = "orp-" + mid_or.split("/")[1]
        MODELS[mid] = ("openrouter-paid", f"openrouter/{mid_or}", None)
        fails["openrouter-paid"] = 0
        try: run_model(mid, tasks, fails)
        except Blocked as e:
            if re.search(r"credit|402|insufficient|limit exceeded", str(e), re.I): log(f"BLOCKED (hesap) {e}"); break
            log(f"  MODEL ATLANDI {mid}: {e}")                  # modele özgü sorun; kuyruk devam eder
        except Exception as e: log(f"HATA {mid}: {e}")
        log(f"  harcama şu ana kadar: ${or_usage() - start:.3f}")
    log("ÜCRETLİ BİTTİ")

if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    if sys.argv[1:] == ["paid"]:
        run_paid(B.load_tasks()); sys.exit()
    tasks = B.load_tasks()
    want = sys.argv[1:]
    fails = {p: 0 for p in ("codex", "opencode", "agy", "muse", "openrouter")}
    ths = [threading.Thread(target=provider_worker, args=(p, tasks, fails, want)) for p in fails]
    for t in ths: t.start()
    for t in ths: t.join()
    log("CLI BENCH BİTTİ")
