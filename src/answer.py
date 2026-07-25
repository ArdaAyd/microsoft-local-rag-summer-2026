"""Cevap üretme: retrieve edilen chunk'lar + soru -> Foundry chat -> Türkçe cevap + kaynak.

Yalnızca bağlamdan cevaplanır; bağlamda yoksa "bilmiyorum" denir. Kaynaklar chunk
etiketlerinden üretilir (LLM'e bırakılmaz, uydurmasın diye).
"""
from openai import OpenAI
from config import FOUNDRY_BASE_URL, FOUNDRY_API_KEY, CHAT_MODEL, TOP_K
from retrieve import retrieve

_client = OpenAI(base_url=FOUNDRY_BASE_URL, api_key=FOUNDRY_API_KEY)

# 2. kural ölçülen bir halüsinasyon için eklendi (retrieval güçlenince model, ilgili görünen
# ama sorulanı cevaplamayan bir değeri kapıyordu). Bkz. docs/EXPERIMENTS.md (Deney 8).
SYSTEM_PROMPT = (
    "Sen Teknofest Robotaksi resmi dokümanları hakkında soru yanıtlayan bir asistansın.\n"
    "KURALLAR:\n"
    "1. SADECE sana verilen BAĞLAM'da açıkça yazan bilgiyi kullan; kendi bilgini ekleme.\n"
    "2. Bir değeri (sayı, model adı, kod) ancak bağlamda SORULAN ŞEYİN karşılığı olarak "
    "açıkça yazıyorsa ver. Konuyla ilgili görünen ama başka bir şeyi anlatan değerleri KULLANMA.\n"
    "3. Bağlam soruyu doğrudan cevaplamıyorsa tahmin etme; aynen şunu yaz: "
    "'Bu bilgi verilen dokümanlarda bulunmuyor.'\n"
    "4. Cevabı Türkçe, kısa ve net yaz."
)


def _format_context(hits):
    blocks = []
    for h in hits:
        tag = f"[{h['doc_name']} | s.{h['page']} | {h['section']}]"
        blocks.append(f"{tag}\n{h['text']}")
    return "\n\n".join(blocks)


def _sources(hits):
    """Kaynakları chunk etiketlerinden üretir (LLM'e bırakmayız -> uydurmasın)."""
    seen = []
    for h in hits:
        s = f"{h['doc_name']} / {h['section']} (s.{h['page']})"
        if s not in seen:
            seen.append(s)
    return seen


def answer(question, top_k=TOP_K, model=CHAT_MODEL):
    hits = retrieve(question, top_k=top_k)
    user_msg = f"BAĞLAM:\n{_format_context(hits)}\n\nSORU: {question}"
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return {
        "answer": resp.choices[0].message.content.strip(),
        "sources": _sources(hits),
        "hits": hits,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Kırmızı ışık ihlalinin ceza puanı nedir?"
    r = answer(q)
    print(f"SORU: {q}\n")
    print("CEVAP:\n" + r["answer"])
    print("\nKAYNAKLAR:")
    for s in r["sources"]:
        print("  - " + s)
