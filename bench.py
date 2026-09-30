#!/usr/bin/env python3
"""Finans LLM benchmark çalıştırıcısı (Tier-1, judge'sız, yerel OpenAI-uyumlu sunucu).
  python3 bench.py run [model_id ...]   models.tsv sırasıyla (ya da verilenler); kaldığı yerden devam eder
  python3 bench.py report               results/summary.csv + results/leaderboard.md
Her model: (hdd ise) Mac'e kopyala + doğrula -> sunucu başlat -> görevler (örnek başına journal) -> sunucuyu kapat -> kopyayı sil.
"""
import os, re, sys, json, csv, time, random, shutil, signal, hashlib, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, STAGE = f"{ROOT}/data", f"{ROOT}/stage"
SMOKE = os.environ.get("BENCH_SMOKE")                # duman testi: görev başına 4 örnek, ayrı klasör
RAW = f"{ROOT}/results/{'smoke' if SMOKE else 'raw'}"
PORT, CONC, SEED = 8090, 4, 1234
N = {"fpb": 200, "fiqasa": 235, "tfns": 200, "finqa": 50, "convfinqa": 50, "tatqa": 50, "finreason": 50}   # Tier-1
if SMOKE: N = {k: 4 for k in N}
THINK_BUDGET = 1024
MIN_TPS = 10.0          # kullanıcı kuralı: tek akışta 10 tok/s altı modeller benchmark'a alınmaz
NUM_INSTR = ("\n\nSolve the question. Keep reasoning short. End your reply with a final line of the form "
             "'Answer: <value>' (a number without units if the answer is numeric).")
SENT_INSTR = "\n\nAnswer with exactly one word: Positive, Negative, or Neutral."
LABELS = ["negative", "neutral", "positive"]

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

# ---------------- veri ----------------
def sample(items, n):
    items = sorted(items, key=lambda x: x["id"])
    return items if len(items) <= n else random.Random(SEED).sample(items, n)

def load_tasks():
    T = {}
    for key, sub in (("fpb", "FPB"), ("fiqasa", "FiQA_SA")):
        d = json.load(open(f"{DATA}/adaptllm/{sub}/test.json"))
        T[key] = sample([{"id": f"{key}{x['id']}", "prompt": x["input"].split("\n\n")[-1] + SENT_INSTR,   # 5-shot değil: sadece son soru
                          "gold": x["options"][x["gold_index"]].lower(), "kind": "sent"} for x in d], N[key])
    tf = pd.read_csv(f"{DATA}/tfns/valid.csv")
    lab = {0: "negative", 1: "positive", 2: "neutral"}          # 0 bearish, 1 bullish, 2 neutral
    T["tfns"] = sample([{"id": f"tfns{i}", "prompt": f"Financial tweet: {r.text}\nWhat is the sentiment of this tweet for investors "
                         "(Positive = bullish, Negative = bearish)?" + SENT_INSTR, "gold": lab[int(r.label)], "kind": "sent"}
                        for i, r in tf.iterrows()], N["tfns"])
    for key in ("finqa", "convfinqa", "tatqa"):
        df = pd.read_parquet(f"{DATA}/{key}/test.parquet")
        T[key] = sample([{"id": str(r.id), "prompt": re.sub(r"\s*Answer:\s*$", "", r.query) + NUM_INSTR,
                          "gold": str(r.answer), "kind": "tat" if key == "tatqa" else "num"} for r in df.itertuples()], N[key])
    fr = json.load(open(f"{DATA}/financereasoning/hard.json"))
    T["finreason"] = sample([{"id": x["question_id"], "prompt": f"{x['context']}\n\nQuestion: {x['question']}" + NUM_INSTR,
                              "gold": str(x["ground_truth"]), "kind": "num"} for x in fr], N["finreason"])
    return T

# ---------------- skorlama ----------------
NUM = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")
def strip_think(t): return re.sub(r"<think>.*?(</think>|$)", "", t or "", flags=re.S).strip()

def to_num(s):
    s = s.strip().replace("$", "").replace(",", "").rstrip(".")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()%")
    try: v = float(s)
    except ValueError: return None
    return -v if neg else v

def final_answer(text):
    m = re.findall(r"answer\s*(?:is)?\s*[:=]\s*\**\s*([^\n]+)", text, re.I)
    return (m[-1] if m else text).strip().strip("*` ")

def last_number(s):
    m = NUM.findall(s)
    return to_num(m[-1]) if m else None

def num_ok(pred, gold):
    if pred is None or gold is None: return False
    for p in (pred, pred / 100, pred * 100):
        if abs(p - gold) <= max(abs(gold) * 0.01, 0.005): return True
    return False

def f1(a, b):
    a, b = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not a or not b: return float(a == b)
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if common == 0: return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)

def score(kind, text, gold):
    t = strip_think(text)
    if kind == "sent":
        m = re.findall(r"\b(positive|negative|neutral|bullish|bearish)\b", final_answer(t).lower())   # son etiket = karar
        pred = {"bullish": "positive", "bearish": "negative"}.get(m[-1], m[-1]) if m else ""
        return pred, float(pred == gold)
    ans = final_answer(t)
    g = to_num(gold)
    if g is not None and "\n" not in gold.strip():
        p = last_number(ans)
        return ans[:80], float(num_ok(p, g))
    if kind == "num":                                        # yes/no gibi
        return ans[:80], float(gold.strip().lower() in ans.lower()[:20])
    return ans[:200], f1(ans, gold)                           # TAT-QA metin/çoklu span: token F1

# ---------------- sunucu ----------------
def start_server(m, path):
    if m["fmt"] == "gguf":
        cmd = ["llama-server", "-m", path, "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-fa", "on",
               "-c", str(8192 * CONC), "-np", str(CONC), "--jinja", "--reasoning-format", "deepseek",
               "--reasoning-budget", str(THINK_BUDGET), "--chat-template-kwargs", '{"enable_thinking": false}', "--no-webui"]
    else:
        cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", path, "--host", "127.0.0.1", "--port", str(PORT),
               "--temp", "0", "--chat-template-args", '{"enable_thinking": false}']
    lf = open(f"{RAW}/{m['id']}/server.log", "a")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(600):                                          # büyük modelin yüklenmesi birkaç dakika sürebilir
        time.sleep(1)
        if p.poll() is not None: raise RuntimeError(f"sunucu kapandı (bkz. {RAW}/{m['id']}/server.log)")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2); return p
        except Exception: pass
    raise RuntimeError("sunucu 10 dk içinde hazır olmadı")

def stop_server(p):
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try: p.wait(30)
        except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL)

def probe_speed(m):
    """Tek istek, 128 token üretim -> tok/s (llama.cpp timings; MLX'te completion_tokens / süre)."""
    body = {"messages": [{"role": "user", "content": "Write a detailed paragraph explaining how a company's balance sheet, income statement and cash flow statement relate to each other."}],
            "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()                      # ısınma
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t0
    t = (r.get("timings") or {}).get("predicted_per_second")
    return t if t else (r.get("usage", {}).get("completion_tokens") or 0) / dt

def ask(m, prompt, kind):
    think = m["think"] == "1"
    max_tok = (256 if kind == "sent" else 1024) + (THINK_BUDGET if think else 0)
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": SEED,
            "max_tokens": max_tok,
            "stop": ["<|endoftext|>", "|endoftext|>", "<|im_end|>", "<|eot_id|>", "</s>"],   # bozuk EOS'lu quant'larda kaçak üretimi keser
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    c = r["choices"][0]
    msg = c["message"]
    text = (msg.get("content") or "")
    if msg.get("reasoning_content") and not text.strip():       # bütçe biterse cevap reasoning içinde kalabilir
        text = msg["reasoning_content"]
    u = r.get("usage", {})
    return dict(text=text, finish=c.get("finish_reason"), tok_in=u.get("prompt_tokens"), tok_out=u.get("completion_tokens"),
                lat=round(time.time() - t0, 2), tps=(r.get("timings") or {}).get("predicted_per_second"))

# ---------------- model hazırlığı ----------------
def sha256(path):
    files = [path] if os.path.isfile(path) else sorted(f"{path}/{f}" for f in os.listdir(path) if f.endswith(".safetensors"))
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            while b := fh.read(16 << 20): h.update(b)
    return h.hexdigest()

def stage(m):
    """hdd modeli Mac'e kopyala (rsync kaldığı yerden sürer), boyutu doğrula. Mac modeli yerinde kullanılır."""
    if m["src"] == "mac": return m["path"]
    dst = f"{STAGE}/{m['id']}" + ("" if m["fmt"] == "mlx" else "/" + os.path.basename(m["path"]))
    src = m["path"] + ("/" if m["fmt"] == "mlx" else "")
    os.makedirs(os.path.dirname(dst) if m["fmt"] == "gguf" else dst, exist_ok=True)
    log(f"  kopyalanıyor ({m['gb']} GB): {m['path']}")
    subprocess.run(["rsync", "-a", "--exclude", "._*", "--exclude", ".cache", src, dst + ("/" if m["fmt"] == "mlx" else "")], check=True)
    size = lambda p: os.path.getsize(p) if os.path.isfile(p) else sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs if not f.startswith("._") and ".cache" not in a)
    if size(m["path"]) != size(dst): raise RuntimeError("kopya boyutu tutmuyor")
    return dst

def unstage(m):
    if m["src"] == "hdd": shutil.rmtree(f"{STAGE}/{m['id']}", ignore_errors=True)   # sadece bu scriptin kopyası

def contended():
    """Ağır dönüşüm süreci var mı (sadece gerçek python/llama-quantize süreçleri; kabuk komut satırları sayılmaz)."""
    for line in subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True).stdout.splitlines():
        exe = os.path.basename(line.split(" ", 1)[0])
        if exe.startswith("llama-quantize") or (exe.lower().startswith("python") and
                                                 (" mlx_lm convert" in line or "convert_hf_to_gguf" in line)):
            return 1
    return 0

# ---------------- çalıştırma ----------------
class Deferred(Exception): pass

def run_model(m, tasks, final=False):
    d = f"{RAW}/{m['id']}"; os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/DONE"): log(f"atlandı (tamam): {m['id']}"); return
    if os.path.exists(f"{d}/SLOW"): log(f"atlandı (yavaş): {m['id']}"); return
    if m["src"] == "hdd" and not os.path.exists(m["path"]): log(f"atlandı (5 TB disk takılı değil): {m['id']}"); return
    if float(m["est_tps"]) < MIN_TPS:
        open(f"{d}/SLOW", "w").write(f"tahmini {m['est_tps']} tok/s < {MIN_TPS}"); log(f"atlandı (tahmini {m['est_tps']} tok/s): {m['id']}"); return
    log(f"=== {m['id']} ({m['fmt']} {m['quant']}, {m['gb']} GB, {m['src']})")
    while float(m["gb"]) > 15 and contended():             # 30B dönüşümü RAM'i doldururken büyük model yükleme
        if not final: raise Deferred()
        log("  dönüştürme sürüyor, büyük model için bekleniyor…"); time.sleep(120)
    path = stage(m)
    meta = f"{d}/meta.json"
    if not os.path.exists(meta):
        json.dump({**m, "sha256": sha256(path), "server": subprocess.run(["llama-server", "--version"], capture_output=True, text=True).stderr.strip().splitlines()[-1]
                   if m["fmt"] == "gguf" else "mlx_lm", "seed": SEED, "temperature": 0}, open(meta, "w"), indent=1)
    srv = None
    try:
        srv = start_server(m, path)
        tps = probe_speed(m)
        log(f"  hız ölçümü: {tps:.1f} tok/s (tek akış)")
        json.dump({"single_stream_tps": round(tps, 1)}, open(f"{d}/speed.json", "w"))
        if tps < MIN_TPS:
            open(f"{d}/SLOW", "w").write(f"ölçülen {tps:.1f} tok/s < {MIN_TPS}"); log(f"  YAVAŞ, atlandı: {m['id']}"); return
        for key, items in tasks.items():
            jp = f"{d}/{key}.jsonl"
            done = set()
            if os.path.exists(jp):
                for l in open(jp):
                    try: done.add(json.loads(l)["id"])
                    except Exception: pass                    # yarım son satır
            todo = [x for x in items if x["id"] not in done]
            if not todo: continue
            log(f"  {key}: {len(done)} tamam, {len(todo)} kaldı")
            with open(jp, "a") as jf, ThreadPoolExecutor(CONC) as ex:
                def one(x):
                    try: r = ask(m, x["prompt"], x["kind"])
                    except Exception as e: r = dict(text="", finish=f"error: {e}", tok_in=None, tok_out=None, lat=None, tps=None)
                    pred, sc = score(x["kind"], r["text"], x["gold"])
                    return {"id": x["id"], "gold": x["gold"], "pred": pred, "score": sc, "contended": contended(), **r}
                for rec in ex.map(one, todo):
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
        open(f"{d}/DONE", "w").write(time.strftime("%F %T"))
        log(f"  tamam: {m['id']}")
    finally:
        stop_server(srv)
        unstage(m)

def models():
    return list(csv.DictReader(open(f"{ROOT}/models.tsv"), delimiter="\t"))

def report():
    rows = []
    ms = models()
    known = {m["id"] for m in ms}
    for extra in sorted(os.listdir(RAW)) if os.path.isdir(RAW) else []:     # CLI (frontier) modelleri
        if extra not in known and os.path.exists(f"{RAW}/{extra}/meta.json"):
            mt = json.load(open(f"{RAW}/{extra}/meta.json"))
            ms.append({"id": extra, "group": mt.get("group", "frontier-cli"), "fmt": "api", "quant": mt.get("model", ""), "gb": ""})
    for m in ms:
        d = f"{RAW}/{m['id']}"
        if not os.path.isdir(d): continue
        row = {"model": m["id"], "group": m["group"], "fmt": m["fmt"], "quant": m["quant"], "gb": m["gb"], "done": os.path.exists(f"{d}/DONE")}
        tps, trunc, n = [], 0, 0
        for key in N:
            jp = f"{d}/{key}.jsonl"
            if not os.path.exists(jp): continue
            recs = list({r["id"]: r for r in (json.loads(l) for l in open(jp) if l.strip())}.values())   # aynı soru iki kez yazıldıysa sonuncusu
            if not recs: continue                                  # boş journal (yarıda kesilen görev)
            row[key] = round(100 * sum(r["score"] for r in recs) / len(recs), 1)
            row[f"{key}_n"] = len(recs)
            tps += [r["tps"] or (r["tok_out"] / r["lat"]) for r in recs if (r.get("tps") or (r.get("tok_out") and r.get("lat")))]
            row["contended_%"] = round(100 * sum(bool(r.get("contended")) for r in recs) / len(recs))
            trunc += sum(r["finish"] == "length" for r in recs); n += len(recs)
        blocks = {"sentiment": ["fpb", "fiqasa", "tfns"], "numeric": ["finqa", "convfinqa", "tatqa", "finreason"]}
        for b, ks in blocks.items():
            v = [row[k] for k in ks if k in row]
            row[b] = round(sum(v) / len(v), 1) if len(v) == len(ks) else None
        if "fpb" not in row: continue                         # ertelenmiş/başlamamış
        row["overall"] = round((row["sentiment"] + row["numeric"]) / 2, 1) if row["sentiment"] is not None and row["numeric"] is not None else None
        row["tok_s"] = round(sorted(tps)[len(tps) // 2], 1) if tps else None
        if os.path.exists(f"{d}/speed.json"): row["tok_s"] = json.load(open(f"{d}/speed.json"))["single_stream_tps"]   # tek akış ölçümü daha doğru
        row["truncated_%"] = round(100 * trunc / n, 1) if n else None
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("overall", ascending=False, na_position="last")
    df.to_csv(f"{ROOT}/results/summary.csv", index=False)
    cols = ["model", "group", "quant", "overall", "sentiment", "numeric", *N.keys(), "tok_s", "truncated_%", "contended_%"]
    with open(f"{ROOT}/results/leaderboard.md", "w") as f:
        f.write(df[[c for c in cols if c in df]].to_markdown(index=False))
    print(df[[c for c in cols if c in df]].to_string(index=False))
    write_html(df, ms)

TASK_META = [  # key, etiket, blok, açıklama, ölçü
    ("fpb", "FPB", "sentiment", "Financial PhraseBank: haber cümlelerinde duygu", "doğruluk"),
    ("fiqasa", "FiQA-SA", "sentiment", "FiQA: hedef şirkete göre mikroblog/başlık duygusu", "doğruluk"),
    ("tfns", "TFNS", "sentiment", "Twitter finans haberleri: bullish / bearish / nötr", "doğruluk"),
    ("finqa", "FinQA", "numeric", "Şirket raporu tablosu + metinden sayısal soru", "sayısal, %1 tolerans"),
    ("convfinqa", "ConvFinQA", "numeric", "Çok turlu sohbette zincirleme sayısal soru", "sayısal, %1 tolerans"),
    ("tatqa", "TAT-QA", "numeric", "Tablo + metin karma soru (sayı veya metin)", "sayısal / token F1"),
    ("finreason", "FinReason", "numeric", "FinanceReasoning zor seviye: çok adımlı hesap", "sayısal, %1 tolerans"),
]

TASK_EN = {"Financial PhraseBank: haber cümlelerinde duygu": "Financial PhraseBank: sentiment in news sentences", "FiQA: hedef şirkete göre mikroblog/başlık duygusu": "FiQA: microblog/headline sentiment by target company", "Twitter finans haberleri: bullish / bearish / nötr": "Twitter financial news: bullish / bearish / neutral", "Şirket raporu tablosu + metinden sayısal soru": "Numerical question from company report table + text", "Çok turlu sohbette zincirleme sayısal soru": "Chained numerical question in multi-turn chat", "Tablo + metin karma soru (sayı veya metin)": "Mixed table + text question (number or text)", "FinanceReasoning zor seviye: çok adımlı hesap": "FinanceReasoning hard level: multi-step calculation", "doğruluk": "accuracy", "sayısal, %1 tolerans": "numeric, 1% tolerance", "sayısal / token F1": "numeric / token F1"}   # ortak translate motoruyla çevrildi (2026-09-30)

def write_html(df, ms):
    """results/leaderboard.html: şablona güncel veriyi gömer (yeni biten modeller otomatik eklenir)."""
    tpl = f"{ROOT}/leaderboard_template.html"
    if not os.path.exists(tpl): return
    rows = []
    for r in df.to_dict("records"):
        def num(k):
            try: v = float(r.get(k))
            except (TypeError, ValueError): return None
            return None if pd.isna(v) else v
        row = {"model": r["model"], "group": r["group"], "quant": str(r.get("quant") or ""), "gb": num("gb"),
               "overall": num("overall"), "sentiment": num("sentiment"), "numeric": num("numeric"),
               "tok_s": None if r["group"] == "frontier-cli" or (r.get("fmt") == "mlx" and not os.path.exists(f"{RAW}/{r['model']}/speed.json")) else num("tok_s"),   # MLX hızı henüz doğru ölçülmüyor
               "trunc": num("truncated_%"), "complete": bool(r.get("done"))}
        for k, *_ in TASK_META:
            row[k] = num(k); row[k + "_n"] = int(r[k + "_n"]) if num(k + "_n") is not None else None
        rows.append(row)
    done = {r["model"] for r in rows}
    slow = lambda i: os.path.exists(f"{RAW}/{i}/SLOW")
    pending = [{"id": m["id"], "slow": slow(m["id"])} for m in ms if m["id"] not in done]
    try:
        from cli_bench import MODELS as CLI
        pending += [{"id": k, "slow": False} for k in CLI if k not in done and k not in {p["id"] for p in pending}]
    except Exception: pass
    data = {"generated": time.strftime("%Y-%m-%d %H:%M"), "rows": rows, "pending": pending,
            "tasks": [{"key": k, "label": l, "block": b, "desc": d, "metric": me, "desc_en": TASK_EN.get(d, d),
                       "metric_en": TASK_EN.get(me, me), "n": N[k]} for k, l, b, d, me in TASK_META]}
    html = open(tpl, encoding="utf-8").read().replace('/*__DATA__*/{"generated":"","tasks":[],"rows":[],"pending":[]}',
                                                     json.dumps(data, ensure_ascii=False))
    with open(f"{ROOT}/results/leaderboard.html.tmp", "w", encoding="utf-8") as f: f.write(html)
    os.replace(f"{ROOT}/results/leaderboard.html.tmp", f"{ROOT}/results/leaderboard.html")
    # GitHub Pages: aynı sayfa, tam HTML iskeletiyle
    os.makedirs(f"{ROOT}/docs", exist_ok=True)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>html{color-scheme:light dark}body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
            + html + '\n</body>\n</html>\n')
    with open(f"{ROOT}/docs/index.html", "w", encoding="utf-8") as f: f.write(page)
    write_readme(rows, data["generated"])

GROUP_EN = {"frontier-cli": "frontier (CLI)", "openrouter-free": "OpenRouter free", "openrouter-paid": "OpenRouter paid",
            "baseline": "open general", "finance": "finance-tuned", "uncensored": "uncensored"}

def write_readme(rows, generated):
    """README.md içindeki <!-- LEADERBOARD:START/END --> arasını güncel tabloyla değiştirir."""
    rp = f"{ROOT}/README.md"
    if not os.path.exists(rp): return
    f1 = lambda v: "–" if v is None else f"{v:.1f}"
    done = sorted([r for r in rows if r["complete"] and r["overall"] is not None], key=lambda r: -r["overall"])
    part = sorted([r for r in rows if not r["complete"] and r["overall"] is not None], key=lambda r: -r["overall"])
    keys = [k for k, *_ in TASK_META]
    head = "| # | Model | Group | Quant | Overall | Sentiment | Numeric | " + " | ".join(l for _, l, *_ in TASK_META) + " | tok/s |"
    sep = "|" + "---|" * (8 + len(keys))
    lines = [f"_Updated {generated} · {len(done)} models complete · 835 questions per model_", "", head, sep]
    for i, r in enumerate(done, 1):
        lines.append(f"| {i} | {r['model']} | {GROUP_EN.get(r['group'], r['group'])} | {r['quant'] if r['group'] not in ('frontier-cli','openrouter-free','openrouter-paid') else 'API'} | "
                     f"**{f1(r['overall'])}** | {f1(r['sentiment'])} | {f1(r['numeric'])} | " + " | ".join(f1(r[k]) for k in keys) +
                     " | " + ("–" if r["tok_s"] is None else f"{r['tok_s']:.0f}") + " |")
    if part:
        lines += ["", f"<details><summary>{len(part)} incomplete runs (not ranked)</summary>", "", head.replace("| # ", "| "), sep[4:]]
        for r in part:
            lines.append(f"| {r['model']} | {GROUP_EN.get(r['group'], r['group'])} | {r['quant']} | {f1(r['overall'])} | {f1(r['sentiment'])} | {f1(r['numeric'])} | "
                         + " | ".join(f1(r[k]) for k in keys) + " | – |")
        lines += ["", "</details>"]
    txt = open(rp, encoding="utf-8").read()
    a, b = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
    txt = txt[:txt.index(a) + len(a)] + "\n" + "\n".join(lines) + "\n" + txt[txt.index(b):]
    with open(rp + ".tmp", "w", encoding="utf-8") as f: f.write(txt)
    os.replace(rp + ".tmp", rp)

if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    if sys.argv[1] == "report": report(); sys.exit()
    tasks = load_tasks()
    want = sys.argv[2:]
    deferred = []
    for m in models():
        if want and m["id"] not in want: continue
        try: run_model(m, tasks)
        except Deferred: log(f"  ertelendi (dönüştürme sürüyor): {m['id']}"); deferred.append(m)
        except Exception as e: log(f"HATA {m['id']}: {e} (tekrar çalıştırınca devam eder)")
    for m in deferred:
        try: run_model(m, tasks, final=True)
        except Exception as e: log(f"HATA {m['id']}: {e} (tekrar çalıştırınca devam eder)")
    log("BENCH BİTTİ")
