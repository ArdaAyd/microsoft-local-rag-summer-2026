"""Retrieval modu karsilastirmasi: dense vs lexical (BM25) vs hybrid (RRF).

Iki soru setinde birden olcer:
  eval_set.json            -> sorular corpus'tan turetildi (dokumanla kelime paylasiyor)
  eval_set_paraphrase.json -> ayni hedefler, GERCEK KULLANICI ifadesiyle yeniden yazildi

Ikinci set asil testtir: dense embedding'in zayif oldugu yer paraphrase'dir; BM25'in
zayif oldugu yer de odur. Hangisinin ne kadar tastigini burada goruruz.

Calistir:  python compare_retrieval_modes.py   (eval/ klasorunden)
"""
import sys
import json
import unicodedata
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import db                       # noqa: E402
from retrieve import retrieve   # noqa: E402
from config import DEFAULT_STRATEGY  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = [
    ("orijinal  (corpus dili)", HERE / "eval_set.json"),
    ("paraphrase (kullanıcı dili)", HERE / "eval_set_paraphrase.json"),
]
# Varsayilan tum modlar; alt kume icin:  python compare_retrieval_modes.py dense rerank
MODES = sys.argv[1:] or ["dense", "lexical", "hybrid", "rerank"]
KS = [3, 5]


def norm(s):
    return unicodedata.normalize("NFKC", s or "").casefold()


def kw_hit(chunk, keywords):
    hay = norm((chunk["text"] or "") + " " + (chunk.get("section") or ""))
    return any(norm(k) in hay for k in keywords)


def olc(questions, mode, conn):
    agg = {f"{kind}@{k}": 0 for k in KS for kind in ("doc", "kw")}
    misses = []
    for e in questions:
        hits = retrieve(e["q"], strategy=DEFAULT_STRATEGY, top_k=max(KS),
                        conn=conn, mode=mode)
        for k in KS:
            topk = hits[:k]
            if any(h["doc_name"] == e["gold_doc"] for h in topk):
                agg[f"doc@{k}"] += 1
            if any(kw_hit(h, e["keywords"]) for h in topk):
                agg[f"kw@{k}"] += 1
        if not any(kw_hit(h, e["keywords"]) for h in hits[:max(KS)]):
            misses.append(e["q"])
    return agg, misses


def main():
    conn = db.connect()
    cols = [f"{kind}@{k}" for k in KS for kind in ("doc", "kw")]
    tum_misses = {}

    for set_adi, path in SETS:
        questions = json.loads(path.read_text(encoding="utf-8"))["questions"]
        n = len(questions)
        print(f"\n### {set_adi} — {n} soru (strategy={DEFAULT_STRATEGY}) ###\n")
        hdr = f"{'mode':10}" + "".join(f"{c:>9}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for mode in MODES:
            agg, misses = olc(questions, mode, conn)
            print(f"{mode:10}" + "".join(f"{agg[c]/n*100:>8.1f}%" for c in cols))
            tum_misses[(set_adi, mode)] = misses

    # Zor sette hangi sorular hangi modda hala kaciyor
    zor = SETS[1][0]
    print(f"\n--- '{zor}' setinde top-5 kw-miss ---")
    for mode in MODES:
        m = tum_misses[(zor, mode)]
        print(f"\n{mode} ({len(m)}):")
        for q in m:
            print("  -", q)

    conn.close()


if __name__ == "__main__":
    main()
