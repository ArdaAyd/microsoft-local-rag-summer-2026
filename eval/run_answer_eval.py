"""Cevap KALITESI eval'i (retrieval degil, uretilen cevabin dogrulugu).

Iki sey olcer:
  1) answerable  -> cevapta beklenen ayirt edici deger geciyor mu? (dogruluk)
  2) unanswerable-> model reddediyor mu? (halusinasyon guvenligi)

Retrieval eval'i (run_eval.py) 'dogru pasaji buldu mu' der; bu script 'kullanicinin
gordugu cevap dogru mu' der. Ikisi FARKLI seylerdir: dogru pasaj gelse bile kucuk bir
model tabloyu yanlis okuyabilir.

Birden fazla chat modelini karsilastirmak icin model ID'lerini arguman ver:
  python run_answer_eval.py
  python run_answer_eval.py qwen2.5-1.5b-instruct-generic-gpu:4 qwen2.5-7b-instruct-generic-gpu:4

NOT: Model Foundry'de YUKLU olmali:  foundry model load <alias>
"""
import os
import sys
import json
import unicodedata
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import CHAT_MODEL, TOP_K as CFG_TOP_K  # noqa: E402
from answer import answer                          # noqa: E402

# Varsayilan set; farklisi icin:  ANSWER_SET=answer_set_paraphrase.json python run_answer_eval.py
SET_PATH = Path(__file__).resolve().parent / os.getenv("ANSWER_SET", "answer_set.json")
# Kac chunk baglama konsun. Varsayilan = config.TOP_K (uretimle ayni kalsin).
# Baska bir degeri denemek icin:  ANSWER_TOPK=3 python run_answer_eval.py <model>
TOP_K = int(os.getenv("ANSWER_TOPK", str(CFG_TOP_K)))

# Modelin "bilmiyorum" dediğini gosteren isaretler (system prompt'taki cumle + varyantlar)
REFUSAL_MARKERS = [
    "bulunmuyor", "bulunmamaktadır", "bilmiyorum", "yer almıyor",
    "mevcut değil", "belirtilmemiş", "bilgi yok",
]


def norm(s):
    return unicodedata.normalize("NFKC", s or "").casefold()


def is_refusal(ans):
    a = norm(ans)
    return any(norm(m) in a for m in REFUSAL_MARKERS)


def hits_expected(ans, expected):
    a = norm(ans)
    return any(norm(e) in a for e in expected)


def eval_model(model, data):
    ans_ok, ref_ok = 0, 0
    fails = []

    for item in data["answerable"]:
        r = answer(item["q"], top_k=TOP_K, model=model)
        ok = hits_expected(r["answer"], item["expected"])
        ans_ok += ok
        if not ok:
            fails.append(("YANLIS", item["q"], f"beklenen={item['expected']}", r["answer"]))

    for item in data["unanswerable"]:
        r = answer(item["q"], top_k=TOP_K, model=model)
        ok = is_refusal(r["answer"])
        ref_ok += ok
        if not ok:
            fails.append(("UYDURDU", item["q"], "reddetmeliydi", r["answer"]))

    return ans_ok, ref_ok, fails


def main():
    data = json.loads(SET_PATH.read_text(encoding="utf-8"))
    models = sys.argv[1:] or [CHAT_MODEL]
    n_ans = len(data["answerable"])
    n_ref = len(data["unanswerable"])

    print(f"Cevaplanabilir: {n_ans} | Cevaplanamaz: {n_ref} | top_k={TOP_K}\n")

    results = []
    for m in models:
        print(f"--- {m} calisiyor... ---")
        ans_ok, ref_ok, fails = eval_model(m, data)
        results.append((m, ans_ok, ref_ok, fails))

    # --- Tablo ---
    print()
    hdr = f"{'model':44}{'dogruluk':>12}{'red':>10}{'toplam':>10}"
    print(hdr)
    print("-" * len(hdr))
    for m, a, r, _ in results:
        total = (a + r) / (n_ans + n_ref) * 100
        print(f"{m:44}{a}/{n_ans} ({a/n_ans*100:.0f}%)".ljust(len(hdr) - 20)
              + f"{r}/{n_ref}".rjust(8) + f"{total:>9.1f}%")

    # --- Hatalar ---
    for m, _, _, fails in results:
        print(f"\n{m} | hatalar ({len(fails)}):")
        if not fails:
            print("  (yok)")
        for kind, q, exp, got in fails:
            print(f"  [{kind}] {q}")
            print(f"          {exp}")
            print(f"          cevap: {' '.join(got.split())[:150]}")


if __name__ == "__main__":
    main()
