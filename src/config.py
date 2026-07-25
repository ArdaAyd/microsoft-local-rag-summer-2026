"""Merkezi ayarlar: modeller, önekler, yollar, retrieval parametreleri."""
import os
from pathlib import Path

# --- Foundry Local (cevap üretimi) ---
# Servis portu her başlatmada değişir; kodu düzenlemeden `export FOUNDRY_PORT=...` ile ez
# (bkz. `foundry service status`).
FOUNDRY_PORT = os.getenv("FOUNDRY_PORT", "62889")
FOUNDRY_BASE_URL = f"http://127.0.0.1:{FOUNDRY_PORT}/v1"
FOUNDRY_API_KEY = "not-needed"          # Foundry kullanmaz; openai istemcisi bir string bekler

# --- Modeller ---
# Embedding yerelde üretilir (bu Foundry kataloğunda embedding modeli yok).
# bge-m3, gerçekçi ifade üzerinde karşılaştırmayla seçildi (bkz. docs/EXPERIMENTS.md).
EMBED_MODEL = "BAAI/bge-m3"
# e5/bge asimetrik önek ister (query:/passage:); bge-m3 istemez. Modeli değiştirince güncelle.
EMBED_PASSAGE_PREFIX = ""
EMBED_QUERY_PREFIX = ""

# Chat modeli — alias değil TAM Foundry model kimliği olmalı (API alias'ı reddeder).
# Kimliği al: curl .../v1/models  (önce `foundry model load ...`).
CHAT_MODEL = "qwen2.5-7b-instruct-generic-gpu:4"

# --- Yollar ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "corpus.db"
PDF_DIR = DATA_DIR

# İşlenecek dokümanlar: (dosya adı, kısa ad). Kısa ad kaynak gösterme ve eval'de kullanılır.
DOCS = [
    ("2026_Robotaksi-Şartnamesi.pdf", "Sartname"),
    ("2026_Araç_Kullanıcı_Dokümanı.pdf", "KullaniciDokumani"),
    ("2026_Araç_Genel_Bilgilendirme.pdf", "GenelBilgilendirme"),
    ("2026_Araç-Mimari.pdf", "Mimari"),
]

# --- Retrieval ---
TOP_K = 5
DEFAULT_STRATEGY = "heading"            # sorgu anında kullanılan chunking stratejisi

# Arama modu: "dense" | "lexical" | "hybrid" | "rerank".
# "hybrid" ve "rerank" uygulandı ama gerçekçi ifadede dense'in altında kaldı
# (bkz. docs/EXPERIMENTS.md, Deney 6 ve 9) — "dense" kalsın.
RETRIEVAL_MODE = "dense"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 30                  # rerank modunda dense aday havuzu boyutu
