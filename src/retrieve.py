"""Retrieval katmani: bir soru -> en benzer top-k chunk.

RAG'in 'R'si (Retrieve). Uc mod destekler:

  dense   : embedding benzerligi. ANLAMI yakalar ("gaz komutu" ~ "throttle"),
            nadir/teknik terimlerde zayiflayabilir.
  lexical : BM25 kelime eslesmesi (lexical.py). Tam terimde guclu
            (RC_THRT_DATA, MIC-770), paraphrase'de zayif.
  hybrid  : ikisinin SIRALARINI Reciprocal Rank Fusion (RRF) ile birlestirir:
                skor(d) = 1/(K + dense_sira) + 1/(K + lexical_sira)
            RRF yalnizca siralari kullanir -> iki farkli skor olceginin
            normalize edilmesi gerekmez, bu yuzden saglamdir.

Dense benzerlik notu: vektorler normalize oldugu icin nokta carpimi = kosinus.
Kucuk corpus -> brute-force yeterli (buyukte vektor indeksi gerekir).
"""
from config import (
    TOP_K, DEFAULT_STRATEGY, RETRIEVAL_MODE, RERANK_CANDIDATES,
)
from embed import embed_query
from lexical import BM25
import db

_RRF_K = 60          # RRF sabiti; literaturdeki standart deger
_bm25_cache = {}     # strateji -> (satir_sayisi, BM25). Re-ingest sonrasi surec yeniden baslatilmali.


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _passage_text(r):
    """BM25 indeksi, embedding ile AYNI metni gorsun (contextual header dahil)."""
    return (r["section"] + "\n" + r["text"]) if r.get("section") else r["text"]


def _get_bm25(strategy, rows):
    hit = _bm25_cache.get(strategy)
    if hit is None or hit[0] != len(rows):
        _bm25_cache[strategy] = (len(rows), BM25([_passage_text(r) for r in rows]))
    return _bm25_cache[strategy][1]


def _ranks(scores):
    """Skor listesi -> her indeks icin 1'den baslayan sira (en yuksek skor = 1)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    rank = [0] * len(scores)
    for pos, i in enumerate(order, start=1):
        rank[i] = pos
    return rank


def retrieve(query, strategy=DEFAULT_STRATEGY, top_k=TOP_K, conn=None, mode=RETRIEVAL_MODE):
    """Soruya en yakin top_k chunk'i dondurur. Her chunk dict'ine 'score' eklenir."""
    own_conn = conn is None
    if own_conn:
        conn = db.connect()
    try:
        rows = db.fetch_all(conn, strategy)
        if not rows:
            return []

        dense = lex = None
        if mode in ("dense", "hybrid", "rerank"):
            qvec = embed_query(query)
            dense = [_dot(qvec, r["embedding"]) for r in rows]
        if mode in ("lexical", "hybrid"):
            lex = _get_bm25(strategy, rows).scores(query)

        if mode == "rerank":
            # 1) dense ile aday havuzu (ucuz)  2) cross-encoder ile yeniden sirala (pahali)
            from rerank import rerank_scores
            idx = sorted(range(len(rows)), key=lambda i: -dense[i])[:RERANK_CANDIDATES]
            cand = [rows[i] for i in idx]
            skorlar = rerank_scores(query, [_passage_text(r) for r in cand])
            for r, s, i in zip(cand, skorlar, idx):
                r["score"] = s
                r["dense_score"] = dense[i]
            cand.sort(key=lambda r: r["score"], reverse=True)
            return cand[:top_k]

        if mode == "dense":
            for r, s in zip(rows, dense):
                r["score"] = s
        elif mode == "lexical":
            for r, s in zip(rows, lex):
                r["score"] = s
        elif mode == "hybrid":
            dr, lr = _ranks(dense), _ranks(lex)
            for i, r in enumerate(rows):
                # RRF: her iki listede de ust siralarda olan chunk kazanir
                r["score"] = 1.0 / (_RRF_K + dr[i]) + 1.0 / (_RRF_K + lr[i])
                r["dense_rank"] = dr[i]
                r["lex_rank"] = lr[i]
        else:
            raise ValueError(f"bilinmeyen retrieval modu: {mode!r}")

        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_k]
    finally:
        if own_conn:
            conn.close()


def _fmt(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        preview = " ".join(h["text"].split())[:100]
        lines.append(
            f"  {i}. [{h['score']:.4f}] {h['doc_name']} | s.{h['page']} | {h['section']}\n"
            f"     {preview}..."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Araca nasıl gaz veririm, hangi mesajı yayınlamam gerek?"
    print(f"SORU: {q}")
    for m in ("dense", "lexical", "hybrid", "rerank"):
        print(f"\n--- mode={m} ---")
        print(_fmt(retrieve(q, mode=m, top_k=3)))
