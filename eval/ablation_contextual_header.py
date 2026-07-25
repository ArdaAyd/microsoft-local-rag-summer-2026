"""Ablasyon: embedding girdisine 'section' (baslik) eklemenin etkisi.

Ayni embedding modeliyle (config'deki EMBED_MODEL) iki varyant karsilastirilir:
  A) sadece text                 -> baslikta duran bilgi vektore GIRMEZ
  B) section + text ("contextual header", uretimde kullanilan)

Neden onemli: heading chunker ayirt edici bilgiyi bazen basliga koyuyor
(or. 'Dingil Açıklığı > 130'); A varyantinda retrieval bunu goremez.

Tahribatsiz: bellekte embed eder, uretim DB'sindeki vektorlere DOKUNMAZ.
Calistir:  python ablation_contextual_header.py   (eval/ klasorunden)
"""
import sys
import json
import unicodedata
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import db  # noqa: E402
from config import EMBED_MODEL, EMBED_PASSAGE_PREFIX, EMBED_QUERY_PREFIX  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_set.json"
KS = [3, 5]
STRATEGY = "heading"   # ablasyon sadece heading'de anlamli (fixed'de section bos)


def norm(s):
    return unicodedata.normalize("NFKC", s).casefold()


def kw_hit(text, section, keywords):
    hay = norm((text or "") + " " + (section or ""))
    return any(norm(k) in hay for k in keywords)


def main():
    from sentence_transformers import SentenceTransformer

    conn = db.connect()
    rows = conn.execute(
        "SELECT doc_name, page, section, text FROM chunks WHERE strategy=?", (STRATEGY,)
    ).fetchall()
    conn.close()
    chunks = [{"doc_name": d, "page": p, "section": s, "text": t} for d, p, s, t in rows]
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    n = len(questions)

    variants = {
        "A) sadece text": [c["text"] for c in chunks],
        "B) section+text": [(c["section"] + "\n" + c["text"]) if c["section"] else c["text"]
                            for c in chunks],
    }

    st = SentenceTransformer(EMBED_MODEL)
    Q = st.encode([EMBED_QUERY_PREFIX + q["q"] for q in questions],
                  normalize_embeddings=True, convert_to_numpy=True)

    print(f"Model: {EMBED_MODEL} | chunk: {len(chunks)} ({STRATEGY}) | soru: {n}\n")
    results = {}
    for label, passages in variants.items():
        P = st.encode([EMBED_PASSAGE_PREFIX + t for t in passages],
                      normalize_embeddings=True, convert_to_numpy=True)
        agg = {f"{kind}@{k}": 0 for k in KS for kind in ("doc", "kw")}
        misses = []
        for qi, e in enumerate(questions):
            order = np.argsort(-(P @ Q[qi]))
            for k in KS:
                topk = order[:k]
                if any(chunks[j]["doc_name"] == e["gold_doc"] for j in topk):
                    agg[f"doc@{k}"] += 1
                if any(kw_hit(chunks[j]["text"], chunks[j]["section"], e["keywords"]) for j in topk):
                    agg[f"kw@{k}"] += 1
            if not any(kw_hit(chunks[j]["text"], chunks[j]["section"], e["keywords"])
                       for j in order[:5]):
                misses.append(f"{e['gold_doc']}: {e['q']}")
        results[label] = (agg, misses)

    cols = [f"{kind}@{k}" for k in KS for kind in ("doc", "kw")]
    hdr = f"{'varyant':18}" + "".join(f"{c:>9}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for label, (agg, _) in results.items():
        print(f"{label:18}" + "".join(f"{agg[c]/n*100:>8.1f}%" for c in cols))

    for label, (_, misses) in results.items():
        print(f"\n{label} | top-5 kw-miss ({len(misses)}):")
        for s in misses:
            print("  -", s)


if __name__ == "__main__":
    main()
