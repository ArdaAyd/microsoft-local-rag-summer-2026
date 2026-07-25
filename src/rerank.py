"""Cross-encoder reranker (rerank modu).

Dense ile getirilen aday havuzunu, soru ve pasajı birlikte puanlayan bir cross-encoder
ile yeniden sıralar (bi-encoder'ın aksine ikisini beraber okur). Pahalı olduğu için
yalnızca aday havuzuna uygulanır. Bu corpus'ta dense'in önüne geçmedi; ölçüm amaçlı
tutuluyor (bkz. docs/EXPERIMENTS.md, Deney 9).
"""
from functools import lru_cache
from config import RERANK_MODEL


@lru_cache(maxsize=2)
def _get_model(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)


def rerank_scores(query: str, passages: list[str], model: str = RERANK_MODEL) -> list[float]:
    """Her pasaj için sorguya uygunluk skoru döner (yüksek = daha uygun)."""
    if not passages:
        return []
    ce = _get_model(model)
    scores = ce.predict([(query, p) for p in passages])
    return [float(s) for s in scores]


if __name__ == "__main__":
    q = "Araca nasıl gaz veririm?"
    ps = [
        "Topic: /beemobs/RC_THRT_DATA Bu mesaj otonom kontrolcüden araç kontrolcüsüne gönderilir.",
        "WiFi Router özellikleri: 2.4GHz ve 5GHz bant desteği.",
        "Dinamik engeller aracın hareketi esnasında yol üzerinde bulunur.",
    ]
    for p, s in zip(ps, rerank_scores(q, ps)):
        print(f"  {s:8.3f}  {p[:60]}")
