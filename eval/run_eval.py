"""Retrieval eval: eval_set.json uzerinde Recall@k olcer.

Iki metrik:
  - doc_recall@k: dogru dokumandan (gold_doc) en az bir chunk ilk k sonucta mi?
                  (kaba metrik: 'dogru dokumana gitti mi')
  - kw_recall@k:  ilk k chunk'tan birinde beklenen anahtar kelime gecti mi?
                  (ince metrik: 'dogru pasaji buldu mu')

Iki chunking stratejisi (fixed, heading) ayni sette karsilastirilir -> Gun 10-12 zemini.

Calistir:  python run_eval.py     (eval/ klasorunden)
"""
import sys
import json
import unicodedata
from pathlib import Path
from collections import Counter

# Bu script eval/ altinda; src/ modullerini import edebilmek icin yola ekle.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import db                       # noqa: E402
from retrieve import retrieve   # noqa: E402

# Varsayilan soru seti; farkli bir set icin:  python run_eval.py eval_set_paraphrase.json
EVAL_PATH = Path(__file__).resolve().parent / (
    sys.argv[1] if len(sys.argv) > 1 else "eval_set.json"
)
DOCS = ["Sartname", "KullaniciDokumani", "GenelBilgilendirme", "Mimari"]
STRATEGIES = ["fixed", "heading"]
KS = [3, 5]


def norm(s):
    """Turkce-guvenli karsilastirma: NFKC + casefold."""
    return unicodedata.normalize("NFKC", s).casefold()


def load_questions():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]


def validate(questions):
    """Ground truth saglamligi: her sorunun anahtari SADECE gold_doc'ta gecmeli."""
    conn = db.connect()
    blob = {
        d: norm(" ".join((t or "") + " " + (s or "") for (t, s) in
                conn.execute("SELECT text, section FROM chunks WHERE doc_name=?", (d,))))
        for d in DOCS
    }
    conn.close()
    bad = []
    for i, e in enumerate(questions, 1):
        hits = [d for d in DOCS if any(norm(k) in blob[d] for k in e["keywords"])]
        if hits != [e["gold_doc"]]:
            bad.append((i, e["gold_doc"], hits))
    return bad


def kw_in_chunk(chunk, keywords):
    hay = norm((chunk["text"] or "") + " " + (chunk.get("section") or ""))
    return any(norm(k) in hay for k in keywords)


def evaluate():
    questions = load_questions()
    n = len(questions)

    bad = validate(questions)
    if bad:
        print("!! GROUND TRUTH UYARISI (duzeltilmeli):")
        for i, g, h in bad:
            print(f"   #{i} gold={g} ama bulundugu={h}")
        print()
    else:
        print(f"Ground truth dogrulandi: {n}/{n} soru tek-dokumana ozgu.\n")

    conn = db.connect()
    max_k = max(KS)
    results = {}
    for strat in STRATEGIES:
        agg = Counter()
        doc_fails = []
        kw_fails = []
        for e in questions:
            hits = retrieve(e["q"], strategy=strat, top_k=max_k, conn=conn)
            for k in KS:
                topk = hits[:k]
                if any(h["doc_name"] == e["gold_doc"] for h in topk):
                    agg[f"doc@{k}"] += 1
                if any(kw_in_chunk(h, e["keywords"]) for h in topk):
                    agg[f"kw@{k}"] += 1
            if not any(h["doc_name"] == e["gold_doc"] for h in hits[:max_k]):
                doc_fails.append((e["q"], e["gold_doc"], hits[0]["doc_name"]))
            if not any(kw_in_chunk(h, e["keywords"]) for h in hits[:max_k]):
                kw_fails.append((e["q"], e["gold_doc"], e["keywords"]))
        results[strat] = (agg, doc_fails, kw_fails)
    conn.close()

    # --- Tablo ---
    print(f"Eval seti: {n} soru | degerler = Recall (%)\n")
    hdr = f"{'metrik':12}" + "".join(f"{s:>12}" for s in STRATEGIES)
    print(hdr)
    print("-" * len(hdr))
    for k in KS:
        for kind in ("doc", "kw"):
            key = f"{kind}@{k}"
            row = f"{key:12}"
            for s in STRATEGIES:
                row += f"{results[s][0][key] / n * 100:>11.1f}%"
            print(row)

    # --- Basarisizlik detaylari (heading stratejisi) ---
    _, doc_fails, kw_fails = results["heading"]
    print("\nheading | top-5'te dogru DOKUMANA hic gitmeyenler:")
    print("  (yok)" if not doc_fails else "")
    for q, gold, got in doc_fails:
        print(f"  - beklenen={gold}, gelen={got}: {q}")

    print("\nheading | top-5'te dogru PASAJI (anahtar kelime) bulamayanlar:")
    print("  (yok)" if not kw_fails else "")
    for q, gold, kws in kw_fails:
        print(f"  - {gold} | {kws}: {q}")


if __name__ == "__main__":
    evaluate()
