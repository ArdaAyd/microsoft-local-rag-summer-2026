"""Embedding modeli karsilastirmasi — IKI soru setinde birden.

Ayni chunk'lar (heading, section+text) uzerinde farkli embedding modellerini yan yana
koyar. Uretim DB'sine / config'e DOKUNMAZ: her modeli BELLEKTE yeniden embed eder.

KARAR METRIGI = paraphrase setindeki kw@5.
Deney 6 gosterdi ki orijinal set (sorular corpus dilinde) yaniltici: orada iyi gorunen
bir degisiklik gercek kullanimda sistemi bozabiliyor. O yuzden iki sette de olcuyoruz
ama karari ZOR sete gore veriyoruz.

e5/bge onek notu: e5 ailesi asimetrik onek ister (query:/passage:); bge-m3 istemez.
Her model kendi onegiyle tanimli.

Calistir:  python compare_embeddings.py                # tum modeller
           python compare_embeddings.py e5-small e5-base   # sadece secilenler (label ile)
"""
import gc
import sys
import json
import unicodedata
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import db  # noqa: E402

HERE = Path(__file__).resolve().parent
DOCS = ["Sartname", "KullaniciDokumani", "GenelBilgilendirme", "Mimari"]
KS = [3, 5]
STRATEGY = "heading"

SETS = [
    ("orijinal", HERE / "eval_set.json"),
    ("paraphrase", HERE / "eval_set_paraphrase.json"),
]

MODELS = [
    {"label": "MiniLM-L12",       "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "qp": "", "pp": ""},
    {"label": "e5-small",         "id": "intfloat/multilingual-e5-small", "qp": "query: ", "pp": "passage: "},
    {"label": "e5-base",          "id": "intfloat/multilingual-e5-base",  "qp": "query: ", "pp": "passage: "},
    {"label": "bge-m3",           "id": "BAAI/bge-m3",                    "qp": "", "pp": ""},
]


def norm(s):
    return unicodedata.normalize("NFKC", s or "").casefold()


def kw_hit(chunk, keywords):
    hay = norm((chunk["text"] or "") + " " + (chunk.get("section") or ""))
    return any(norm(k) in hay for k in keywords)


def fetch_chunks():
    conn = db.connect()
    rows = conn.execute(
        "SELECT doc_name, page, section, text FROM chunks WHERE strategy=?", (STRATEGY,)
    ).fetchall()
    conn.close()
    return [{"doc_name": d, "page": p, "section": s, "text": t} for d, p, s, t in rows]


def main():
    from sentence_transformers import SentenceTransformer

    secilen = set(sys.argv[1:])
    modeller = [m for m in MODELS if not secilen or m["label"] in secilen]

    chunks = fetch_chunks()
    passages = [(c["section"] + "\n" + c["text"]) if c["section"] else c["text"] for c in chunks]
    setler = [(ad, json.loads(p.read_text(encoding="utf-8"))["questions"]) for ad, p in SETS]

    print(f"Chunk: {len(chunks)} ({STRATEGY}) | modeller: {[m['label'] for m in modeller]}\n")
    sonuc = {}   # (label, set_adi) -> agg

    for m in modeller:
        print(f"--- {m['label']} yukleniyor/olculuyor... ---", flush=True)
        st = SentenceTransformer(m["id"])
        P = st.encode([m["pp"] + t for t in passages],
                      normalize_embeddings=True, convert_to_numpy=True)
        for set_adi, questions in setler:
            Q = st.encode([m["qp"] + q["q"] for q in questions],
                          normalize_embeddings=True, convert_to_numpy=True)
            agg = {f"{kind}@{k}": 0 for k in KS for kind in ("doc", "kw")}
            for qi, e in enumerate(questions):
                order = np.argsort(-(P @ Q[qi]))
                for k in KS:
                    topk = order[:k]
                    if any(chunks[j]["doc_name"] == e["gold_doc"] for j in topk):
                        agg[f"doc@{k}"] += 1
                    if any(kw_hit(chunks[j], e["keywords"]) for j in topk):
                        agg[f"kw@{k}"] += 1
            sonuc[(m["label"], set_adi)] = (agg, len(questions))
        # bellegi bosalt (16GB'ta buyuk modellerle sikisiyoruz)
        del st, P
        gc.collect()

    cols = [f"{kind}@{k}" for k in KS for kind in ("doc", "kw")]
    for set_adi, _ in setler:
        print(f"\n### {set_adi} seti ###\n")
        hdr = f"{'model':16}" + "".join(f"{c:>9}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for m in modeller:
            agg, n = sonuc[(m["label"], set_adi)]
            print(f"{m['label']:16}" + "".join(f"{agg[c]/n*100:>8.1f}%" for c in cols))

    print("\n### KARAR METRIGI: paraphrase kw@5 ###")
    for m in modeller:
        agg, n = sonuc[(m["label"], "paraphrase")]
        print(f"  {m['label']:16} {agg['kw@5']/n*100:>6.1f}%")


if __name__ == "__main__":
    main()
