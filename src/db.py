"""SQLite katmanı. Her chunk kaynak etiketiyle (doc_name, page, section) saklanır.

  - doc_name/page/section: hem kaynak gösterme hem "doğru dokümana gitti mi" ölçümü için.
  - strategy: aynı DB'de birden fazla chunking'i tutup karşılaştırmak için.
  - embedding: JSON string. Küçük corpus'ta brute-force cosine yeterli (büyükte vektör indeksi).
"""
import sqlite3
import json
from config import DB_PATH


def connect():
    return sqlite3.connect(DB_PATH)


def init(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY,
            doc_name  TEXT NOT NULL,
            page      INTEGER,
            section   TEXT,
            strategy  TEXT NOT NULL,
            text      TEXT NOT NULL,
            embedding TEXT            -- JSON list[float]
        )
    """)
    conn.commit()


def clear_strategy(conn, strategy):
    """Bir stratejiyi yeniden ingest etmeden önce eski kayıtları sil."""
    conn.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))
    conn.commit()


def insert_chunk(conn, c, embedding):
    conn.execute(
        "INSERT INTO chunks (doc_name, page, section, strategy, text, embedding) "
        "VALUES (?,?,?,?,?,?)",
        (c["doc_name"], c["page"], c["section"], c["strategy"],
         c["text"], json.dumps(embedding)),
    )


def fetch_all(conn, strategy):
    """Bir stratejinin tüm chunk'larını (embedding'leriyle) getir."""
    cur = conn.execute(
        "SELECT id, doc_name, page, section, text, embedding "
        "FROM chunks WHERE strategy = ?", (strategy,)
    )
    rows = []
    for id_, doc, page, section, text, emb in cur.fetchall():
        rows.append({
            "id": id_, "doc_name": doc, "page": page, "section": section,
            "text": text, "embedding": json.loads(emb),
        })
    return rows


def counts(conn):
    cur = conn.execute(
        "SELECT strategy, doc_name, COUNT(*) FROM chunks GROUP BY strategy, doc_name"
    )
    return cur.fetchall()
