"""Lexical (kelime tabanlı) arama — BM25. "hybrid"/"lexical" modlarında kullanılır.

BM25 formülü:
    score(q,d) = SUM_t  IDF(t) * f(t,d)*(k1+1) / ( f(t,d) + k1*(1 - b + b*|d|/avgdl) )
    IDF(t)     = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )
  f(t,d): terim frekansı | df(t): terimi içeren doküman sayısı | avgdl: ortalama uzunluk
  k1 terim frekansı doygunluğunu, b uzunluk normalizasyonunu ayarlar.

Türkçe için aksanlar kaldırılır (PDF çıkarımı tutarsız olabildiği için 'Açıklığı'/'Acikligi'
eşleşsin). Stemming yok.
"""
import math
import re
import unicodedata
from collections import Counter

K1 = 1.5
B = 0.75
_TOKEN_RE = re.compile(r"[0-9a-z]+")


def _sadelestir(text: str) -> str:
    """NFKD -> birlesen isaretleri (aksan) at -> casefold. Turkce'yi ASCII'ye indirger."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.casefold()


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(_sadelestir(text)) if len(w) > 1]


class BM25:
    def __init__(self, docs: list[str]):
        self.tokens = [tokenize(d) for d in docs]
        self.N = len(self.tokens)
        self.lens = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.lens) / self.N) if self.N else 0.0
        self.tf = [Counter(t) for t in self.tokens]

        df = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        self.idf = {w: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        out = [0.0] * self.N
        if not self.avgdl:
            return out
        for w in tokenize(query):
            idf = self.idf.get(w)
            if idf is None:          # corpus'ta hic gecmeyen kelime -> katkisiz
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.lens[i] / self.avgdl)
                out[i] += idf * f * (K1 + 1) / denom
        return out


if __name__ == "__main__":
    bm = BM25([
        "Topic: /beemobs/RC_THRT_DATA gaz komutu",
        "Advantech MIC-770V3H Endüstriyel Box PC",
        "Dinamik engeller yolda hareketli çıkar",
    ])
    for q in ["RC_THRT_DATA", "MIC-770", "dinamik engel"]:
        print(f"{q!r} -> {[round(s, 2) for s in bm.scores(q)]}")
