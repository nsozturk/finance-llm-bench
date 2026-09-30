#!/usr/bin/env python3
"""Benchmark edilecek modellerin listesi -> models.tsv (en hızlıdan en yavaşa).
  python3 registry.py
Kaynaklar: Mac LM Studio klasörü (src=mac, yerinde çalışır, silinmez) + ns-5tb (src=hdd, Mac'e kopyalanır, test sonrası kopya silinir).
"""
import os, re, glob, json

HDD = "/Volumes/ns-5tb/AI-Models"
FIN = f"{HDD}/Finance-LLMs"
LMS = os.path.expanduser("~/.lmstudio/models")
BW = 240.0  # GB/s, M1 Max pratik bant genişliği (model_fit_m1max.py ile aynı varsayım)

# Finance-LLMs içinde atlananlar: GGUF'u Mac'te olan tam hassasiyet kopyalar, LLM olmayanlar, tabanı olmayan LoRA'lar
SKIP_FIN = re.compile(r"^(AdaptLLM|DianJin--DianJin-R1-7B|SUFE-AIFLM-Lab--Fin-R1|TheFinAI--Fin-o1-14B|TheFinAI--Fino1-8B|"
                      r"OpenDataArena--ODA-Fin-RL-8B|Josephgflowers--Phinance|rLLM--rLLM-FinQA-4B$|Kronos|FinGPT--|0xZaid10|ProsusAI|_)")
FIN_BF16 = ["Alpha-R1", "TheFinAI__Fino1-14B"]            # GGUF'u olmayan, bf16 olarak MLX ile koşulacaklar
BASELINES = [                                             # genel amaçlı açık modeller (kıyas)
    "Alibaba Qwen/normal/Qwen3.5-4B", "Alibaba Qwen/normal/Qwen3.5-9B", "Alibaba Qwen/normal/Qwen3.5-27B-4bit",
    "Alibaba Qwen/normal/Qwen3.5-35B-A3B-4bit", "Alibaba Qwen/normal/Qwen3.8-27B-oQ4e-mtp",
    "Google/normal/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    # K2-Horizon 7B/36B: brew llama.cpp (build 10964) 'k2-horizon' mimarisini tanımıyor -> çıkarıldı (2026-09-30)
    "Zhipu AI/normal/GLM-4.7-Flash-MLX-6bit", "NVIDIA/normal/NVIDIA-Nemotron-Nano-12B-v2.Q4_K_M.gguf",
    "Allen AI/normal/Olmo-3-7B-Think.i1-Q4_K_M.gguf",
    "Google/normal/google_gemma-4-E2B-it-GGUF/google_gemma-4-E2B-it-Q4_K_M.gguf",
    "Google/normal/google_gemma-4-E4B-it-GGUF/google_gemma-4-E4B-it-Q4_K_M.gguf",
    "Google/normal/google_gemma-4-31B-it-GGUF/google_gemma-4-31B-it-Q4_K_M.gguf",
    "OpenAI/normal/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
    "Alibaba Qwen/normal/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
]
PRIORITY = [   # kullanıcı isteği (2026-09-30): Mac testlerinde önce bunlar
    ("Google/normal/google_gemma-4-31B-it-GGUF/google_gemma-4-31B-it-Q4_K_M.gguf", "baseline"),
    ("Alibaba Qwen/abliterated/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/Qwen3.8-27B-Heretic-Q4_K_M.gguf", "uncensored"),
    ("Alibaba Qwen/abliterated/Qwen3.8-27B-Fable-Distill-Heretic-ara-GGUF/Qwen3.8-27B-Fable-Distill-Heretic-ara-Q4_K_S.gguf", "uncensored"),
    ("Alibaba Qwen/abliterated/Huihui-Qwen3.8-27B-abliterated-GGUF__ns-1tb-ekranlı/Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf", "uncensored"),
    ("Alibaba Qwen/abliterated/Qwen3.8-27B-ABLITERATED-GGUF/Qwen3.8-27B-ABLITERATED-Q8_0.gguf", "uncensored"),
    ("Alibaba Qwen/abliterated/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf", "uncensored"),
]
THINK = re.compile(r"R1|o1|Fino1|QwQ|Think|reason|ODA-Fin-RL|Open-Finance-R|rLLM|DeepHermes|cpa-qwen3|DianJin", re.I)

def size_gb(p):
    if os.path.isfile(p): return os.path.getsize(p) / 1e9
    return sum(os.path.getsize(f) for f in glob.glob(f"{p}/*.safetensors")) / 1e9

def active_ratio(name):
    if "gpt-oss-20b" in name.lower(): return 3.6 / 21
    if "GLM-4.7-Flash" in name: return 3 / 30
    m = re.search(r"(\d+(?:\.\d+)?)B-A(\d+(?:\.\d+)?)B", name, re.I)
    return float(m.group(2)) / float(m.group(1)) if m else 1.0

def mlx_quant(p):
    try: return json.load(open(f"{p}/config.json")).get("quantization", {}).get("bits")
    except Exception: return None

def entry(path, src, group):
    name = os.path.basename(path).removesuffix(".gguf")
    parent = os.path.basename(os.path.dirname(path))
    if path.endswith(".gguf") and parent not in ("normal", "abliterated", "gguf"):   # dosya adı genelde jenerik (gemma-4-E4B-it.Q4_K_M) -> repo klasör adı
        name = re.sub(r"[-_]gguf$", "", parent, flags=re.I)
    elif parent == "gguf":
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    fmt = "gguf" if path.endswith(".gguf") else "mlx"
    gb = size_gb(path)
    tps = BW / max(gb * active_ratio(name + path), 0.3)
    think = bool(THINK.search(name))
    cost = (1 + 2 * think) / tps          # sıralama anahtarı: düşünen modeller ~3x daha uzun sürer
    return dict(id=re.sub(r"[^A-Za-z0-9._-]+", "_", name), group=group, src=src, fmt=fmt,
                quant=(re.search(r"(I?Q\d_[\w]*|MXFP4)", os.path.basename(path), re.I) or [None, "?"])[0].upper() if fmt == "gguf" else f"{mlx_quant(path) or 'bf16'}bit",
                gb=round(gb, 1), est_tps=round(tps), think=int(think), cost=cost, path=path)

GENERAL = {"Llama-3.1-8B-Instruct", "Qwen2.5-32B-Instruct", "Qwen_QwQ-32B"}   # Finance-LLMs'te duran ama genel modeller
rows = []
for g in glob.glob(f"{LMS}/**/*.gguf", recursive=True):             # Mac'teki 19 + gated GGUF'lar
    if "mmproj" not in g.lower(): rows.append(entry(g, "mac", "finance"))
for d in sorted(os.listdir(FIN)):
    p = f"{FIN}/{d}"
    if SKIP_FIN.match(d) or not os.path.isdir(p): continue
    ggufs = [g for g in glob.glob(f"{p}/**/*.gguf", recursive=True) if "mmproj" not in g.lower()]
    if ggufs: rows.append(entry(sorted(ggufs)[0], "hdd", "finance"))
    elif os.path.exists(f"{p}/config.json") and (mlx_quant(p) or d in FIN_BF16): rows.append(entry(p, "hdd", "finance"))
for b in BASELINES:
    if os.path.exists(f"{HDD}/{b}"): rows.append(entry(f"{HDD}/{b}", "hdd", "baseline"))   # hf dosyayı ancak tamamlanınca yerine koyar

for r in rows:
    if r["id"] in GENERAL: r["group"] = "baseline"
prio = []
for pth, grp in PRIORITY:
    if os.path.exists(f"{HDD}/{pth}"):
        e = entry(f"{HDD}/{pth}", "hdd", grp)
        if grp == "uncensored" and not e["id"].startswith("Huihui"): e["id"] = e["id"].removesuffix("-GGUF")
        if "__" in e["id"]: e["id"] = e["id"].split("__")[0]
        prio.append(e)
seen = set(); out = []
for r in prio + sorted(rows, key=lambda r: r["cost"]):
    if r["id"] in seen: continue
    seen.add(r["id"]); out.append(r)
cols = ["id", "group", "src", "fmt", "quant", "gb", "est_tps", "think", "path"]
with open("models.tsv.tmp", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in out: f.write("\t".join(str(r[c]) for c in cols) + "\n")
os.replace("models.tsv.tmp", "models.tsv")
print(f"{len(out)} model -> models.tsv  (finance {sum(r['group']=='finance' for r in out)}, baseline {sum(r['group']=='baseline' for r in out)})")
