"""Embedding katmanı (sentence-transformers, yerel/offline).

Foundry kataloğunda embedding modeli olmadığı için embedding'ler yerelde üretilir; Foundry
yalnızca cevap üretimi için kullanılır. e5/bge gibi modeller doküman ve sorgu için farklı
önek ister (config.py), bu yüzden embed_passages / embed_query ayrı tutulur.
"""
from functools import lru_cache
from config import EMBED_MODEL, EMBED_PASSAGE_PREFIX, EMBED_QUERY_PREFIX


@lru_cache(maxsize=4)
def _get_model(name: str):
    # Ağır importu fonksiyona alıyoruz: model yalnızca gerçekten kullanılınca yüklensin.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed_many(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Önek eklemeden metinleri vektöre çevirir (birim normlu -> kosinüs = nokta çarpımı)."""
    m = _get_model(model)
    vecs = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vecs]


def embed_passages(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Doküman/pasaj embedding'i (passage öneki ile)."""
    return embed_many([EMBED_PASSAGE_PREFIX + t for t in texts], model)


def embed_query(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Sorgu embedding'i (query öneki ile)."""
    return embed_many([EMBED_QUERY_PREFIX + text], model)[0]


def embed_one(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Öneksiz tek metin (test/genel amaçlı)."""
    return embed_many([text], model)[0]


if __name__ == "__main__":
    v = embed_query("Kırmızı ışıkta durmanın puanı nedir?")
    print(f"Model: {EMBED_MODEL} | boyut: {len(v)}")
    print("İlk 5 değer:", v[:5])
