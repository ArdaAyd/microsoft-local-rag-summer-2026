"""Ingestion hattı: PDF -> sayfa -> chunk -> embedding -> SQLite.

Çalıştır:  python ingest.py            (her iki strateji)
           python ingest.py heading   (tek strateji)
"""
import sys
from config import PDF_DIR, DOCS
from pdf_extract import extract_pages
from chunker import STRATEGIES
from embed import embed_passages
import db


def _embed_input(c):
    """Embedding girdisi olarak section + text döner ("contextual header").

    heading chunker ayırt edici bilgiyi bazen başlığa koyar (örn. "Dingil Açıklığı > 130");
    başlığı da embedlemek bu bilgiyi vektöre taşır. Saklanan 'text' değişmez.
    Etkisi ölçüldü: bkz. docs/EXPERIMENTS.md (Deney 3).
    """
    sec = (c.get("section") or "").strip()
    return f"{sec}\n{c['text']}" if sec else c["text"]


def ingest_strategy(conn, strategy_name):
    fn = STRATEGIES[strategy_name]
    db.clear_strategy(conn, strategy_name)

    all_chunks = []
    for fname, short in DOCS:
        pages = extract_pages(str(PDF_DIR / fname))
        chunks = fn(pages, short)
        all_chunks.extend(chunks)
        print(f"  {short}: {len(pages)} sayfa -> {len(chunks)} chunk")

    print(f"  [{strategy_name}] toplam {len(all_chunks)} chunk, embedding üretiliyor...")
    BATCH = 32
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        vecs = embed_passages([_embed_input(c) for c in batch])
        for c, v in zip(batch, vecs):
            db.insert_chunk(conn, c, v)
        conn.commit()
        print(f"    {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}", end="\r")
    print(f"\n  [{strategy_name}] tamam.")


def main():
    strategies = sys.argv[1:] or list(STRATEGIES.keys())
    conn = db.connect()
    db.init(conn)
    for s in strategies:
        print(f"=== Strateji: {s} ===")
        ingest_strategy(conn, s)
    print("\n=== DB özeti ===")
    for strategy, doc, n in db.counts(conn):
        print(f"  {strategy:8} | {doc:20} | {n} chunk")
    conn.close()


if __name__ == "__main__":
    main()
