"""Sorgu genişletme (query expansion).

Yerel LLM'den soruya ait teknik terimler alıp sorguya ekler; amaç kullanıcı ifadesi ile
doküman terminolojisi arasındaki kelime uyuşmazlığını köprülemek. Ölçümde retrieval'ı
iyileştirmedi; deney amaçlı tutuluyor (bkz. docs/EXPERIMENTS.md, Deney 10).

Not: Prompt örnekleri eval setinde geçmeyen kelimelerden seçildi (sonucu şişirmemek için).
"""
from openai import OpenAI
from config import FOUNDRY_BASE_URL, FOUNDRY_API_KEY, CHAT_MODEL

_client = OpenAI(base_url=FOUNDRY_BASE_URL, api_key=FOUNDRY_API_KEY)

EXPAND_PROMPT = (
    "Görevin soruyu CEVAPLAMAK DEĞİL. Bu soru, Türkçe yazılmış otonom araç yarışması "
    "dokümanlarında (şartname, kullanıcı kılavuzu, donanım listesi) aranacak. Sorudaki "
    "gündelik ifadelerin dokümanlarda geçmesi muhtemel TÜRKÇE teknik/resmi eş anlamlılarını "
    "ve gerekiyorsa İngilizce teknik karşılıklarını üret; 4-8 terim "
    "(örnek: sürat → hız, speed; el freni → park freni, parking brake). "
    "SADECE virgülle ayrılmış terimleri yaz. Alt çizgi kullanma. Türkçe ve İngilizce dışında "
    "dil kullanma. Açıklama, cümle, cevap yazma."
)


def expand_terms(question: str, model: str = CHAT_MODEL) -> str:
    """Soru için virgülle ayrılmış teknik terim listesi döndürür."""
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXPAND_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,     # tekrarlanabilirlik: aynı soru -> aynı genişletme
        max_tokens=80,
    )
    terms = " ".join(resp.choices[0].message.content.split())
    return terms[:300]       # taşan gevezeliği kes; embedding'i domine etmesin


def expand_query(question: str, model: str = CHAT_MODEL) -> str:
    """Genişletilmiş sorgu: orijinal soru + terim listesi (soru anlamı korunur)."""
    terms = expand_terms(question, model)
    return f"{question}\n{terms}" if terms else question


if __name__ == "__main__":
    for q in [
        "Aracı kumandayla nasıl sürerim?",
        "Lazer tarayıcının kaç kanalı var?",
        "Araçta hangi bilgisayar kullanılıyor?",
    ]:
        print(f"SORU : {q}")
        print(f"TERİM: {expand_terms(q)}\n")
