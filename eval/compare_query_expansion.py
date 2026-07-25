"""Deney 10: Sorgu genişletme (query expansion) ölçümü.

Her soru için yerel LLM'den teknik terim listesi alınır, sorgunun sonuna eklenir,
dense retrieval genişletilmiş sorguyla yapılır. Baseline = genişletmesiz dense.

KARAR METRİĞİ = paraphrase setinde kw@5 (Deney 6'dan beri kural bu).
Orijinal set regresyon kontrolü.

Kullanım (eval/ klasöründen; Foundry açık ve model yüklü olmalı):
    python compare_query_expansion.py eval_set_paraphrase.json <model-id>
    python compare_query_expansion.py eval_set.json <model-id>
"""
import sys
import json
import time
import unicodedata
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import db                        # noqa: E402
from retrieve import retrieve    # noqa: E402
from expand import expand_terms  # noqa: E402
from config import DEFAULT_STRATEGY, CHAT_MODEL  # noqa: E402

KS = [3, 5]


def norm(s):
    return unicodedata.normalize("NFKC", s or "").casefold()


def kw_hit(c, kws):
    hay = norm((c["text"] or "") + " " + (c.get("section") or ""))
    return any(norm(k) in hay for k in kws)


def main():
    set_file = sys.argv[1] if len(sys.argv) > 1 else "eval_set_paraphrase.json"
    model = sys.argv[2] if len(sys.argv) > 2 else CHAT_MODEL
    questions = json.loads((Path(__file__).parent / set_file).read_text(encoding="utf-8"))["questions"]
    n = len(questions)

    conn = db.connect()
    agg = {v: {f"{kind}@{k}": 0 for k in KS for kind in ("doc", "kw")}
           for v in ("baseline", "expanded")}
    kacanlar = {"baseline": [], "expanded": []}
    sure_toplam = 0.0
    ornekler = []

    for e in questions:
        t0 = time.perf_counter()
        terms = expand_terms(e["q"], model=model)
        sure_toplam += time.perf_counter() - t0
        genis = f"{e['q']}\n{terms}" if terms else e["q"]
        ornekler.append((e["q"], terms))

        for variant, sorgu in (("baseline", e["q"]), ("expanded", genis)):
            hits = retrieve(sorgu, strategy=DEFAULT_STRATEGY, top_k=max(KS),
                            conn=conn, mode="dense")
            for k in KS:
                topk = hits[:k]
                if any(h["doc_name"] == e["gold_doc"] for h in topk):
                    agg[variant][f"doc@{k}"] += 1
                if any(kw_hit(h, e["keywords"]) for h in topk):
                    agg[variant][f"kw@{k}"] += 1
            if not any(kw_hit(h, e["keywords"]) for h in hits[:max(KS)]):
                kacanlar[variant].append(e["q"])
    conn.close()

    cols = [f"{kind}@{k}" for k in KS for kind in ("doc", "kw")]
    print(f"\n### {set_file} — {n} soru | genişletici: {model} ###")
    print(f"Ortalama genişletme süresi: {sure_toplam/n:.1f} sn/soru\n")
    hdr = f"{'varyant':10}" + "".join(f"{c:>9}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for v in ("baseline", "expanded"):
        print(f"{v:10}" + "".join(f"{agg[v][c]/n*100:>8.1f}%" for c in cols))

    duzelen = [q for q in kacanlar["baseline"] if q not in kacanlar["expanded"]]
    bozulan = [q for q in kacanlar["expanded"] if q not in kacanlar["baseline"]]
    print(f"\nGenişletmeyle DÜZELEN ({len(duzelen)}):")
    for q in duzelen:
        print("  +", q)
    print(f"Genişletmeyle BOZULAN ({len(bozulan)}):")
    for q in bozulan:
        print("  -", q)

    print("\nÖrnek genişletmeler (ilk 5):")
    for q, t in ornekler[:5]:
        print(f"  {q}\n    -> {t}")


if __name__ == "__main__":
    main()
