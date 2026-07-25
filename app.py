"""Streamlit arayuzu — Robotaksi RAG.

Calistir (proje kokunden):
    streamlit run app.py

Foundry Local cevap uretimi icin calisir durumda olmali:
    foundry service status      # kapaliysa: foundry service start
    foundry model load qwen2.5-1.5b
    export FOUNDRY_PORT=<status'taki port>

Foundry kapaliyken de "Sadece retrieval" modu calisir (embedding yereldir).
"""
import sys
import json
import time
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Robotaksi RAG", layout="wide")

from config import (  # noqa: E402
    CHAT_MODEL, EMBED_MODEL, FOUNDRY_BASE_URL, TOP_K, DEFAULT_STRATEGY,
)
from retrieve import retrieve  # noqa: E402
from answer import answer      # noqa: E402

# Deney 5'te olculen taka: dogruluk <-> hiz (bkz. README Experiment 5)
MODEL_SECENEKLERI = {
    "qwen2.5-7b — doğru ama yavaş (~27 sn)": "qwen2.5-7b-instruct-generic-gpu:4",
    "qwen2.5-1.5b — hızlı ama daha az doğru (~7 sn)": "qwen2.5-1.5b-instruct-generic-gpu:4",
}

# (kisa etiket, tam soru) — etiket butonda, tam soru tooltip'te
ORNEK_SORULAR = [
    ("Gaz komutu",      "Araca gaz/throttle komutu hangi ROS topic'i ile gönderilir?"),
    ("Dingil açıklığı", "Aracın dingil açıklığı en az kaç olmalı?"),
    ("Dinamik engel",   "Dinamik engel nedir, araç ne yapmalı?"),
    ("LiDAR sensörü",   "Araçta hangi LiDAR sensörü kullanılıyor?"),
    ("Hava yastığı *",  "Araçta kaç adet hava yastığı var?"),  # corpus'ta YOK
]

# Modelin reddettigini anlamak icin (run_answer_eval.py ile ayni mantik)
RED_ISARETLERI = ["bulunmuyor", "bulunmamaktadır", "bilmiyorum", "yer almıyor",
                  "mevcut değil", "bilgi yok"]


def foundry_durumu(model):
    """Foundry'ye ulasilabiliyor mu ve secili model yuklu mu?"""
    try:
        with urllib.request.urlopen(FOUNDRY_BASE_URL.rstrip("/") + "/models", timeout=3) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
        return (model in ids), ids
    except Exception as e:
        return None, str(e)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Ayarlar")
    strategy = st.selectbox(
        "Chunking stratejisi", ["heading", "fixed"],
        index=0 if DEFAULT_STRATEGY == "heading" else 1,
        help="Eval'de ikisi de aynı skoru veriyor; heading okunabilir bölüm etiketi üretir.",
    )
    top_k = st.slider("Kaç chunk getirilsin (top-k)", 1, 8, TOP_K,
                      help="Eval: kw_recall@3=%87.0, kw_recall@5=%91.3")
    sadece_retrieval = st.checkbox(
        "Sadece retrieval (LLM'siz)", value=False,
        help="Foundry kapalıyken de çalışır — embedding tamamen yereldir.",
    )

    st.divider()
    # Varsayilan: config'deki CHAT_MODEL hangisiyse o secili gelsin
    secenekler = list(MODEL_SECENEKLERI)
    varsayilan = next((i for i, k in enumerate(secenekler)
                       if MODEL_SECENEKLERI[k] == CHAT_MODEL), 0)
    secim = st.selectbox(
        "Cevap modeli", secenekler, index=varsayilan,
        help="Deney 5: 7b @ top_k=5 -> doğruluk %100 / reddetme %100; "
             "1.5b @ top_k=5 -> %90 / %75 ama ~4x hızlı.",
    )
    chat_model = MODEL_SECENEKLERI[secim]

    st.caption(f"Embedding (yerel): `{EMBED_MODEL}`")

    yuklu, bilgi = foundry_durumu(chat_model)
    if yuklu is True:
        st.success("Foundry: bağlı, model yüklü")
    elif yuklu is False:
        st.warning("Foundry bağlı ama bu model yüklü değil")
        st.caption(f"Sunucudaki modeller: {bilgi}")
    else:
        st.error("Foundry'ye ulaşılamıyor")
        st.caption(f"{FOUNDRY_BASE_URL}")

# ---------------- Ana ekran ----------------
st.title("Robotaksi RAG")
st.markdown(
    "Teknofest 2026 Robotaksi dokümanlarına soru sor — cevap **yalnızca dokümanlardan** "
    "üretilir ve **kaynak gösterilir**. Dokümanlarda yoksa sistem *bilmiyorum* der."
)

if "soru" not in st.session_state:
    st.session_state.soru = ""

st.caption("Örnek sorular:")
kolonlar = st.columns(len(ORNEK_SORULAR))
for kol, (etiket, tam_soru) in zip(kolonlar, ORNEK_SORULAR):
    with kol:
        if st.button(etiket, help=tam_soru, use_container_width=True):
            st.session_state.soru = tam_soru
st.caption("\\* Bu soru dokümanlarda bilerek yok — sistem uydurmak yerine *bilmiyorum* demeli.")

soru = st.text_input("Sorunuz", value=st.session_state.soru,
                     placeholder="örn. Şerit ihlali nasıl tanımlanır?")

if soru:
    if sadece_retrieval:
        with st.spinner("Getiriliyor…"):
            hits = retrieve(soru, strategy=strategy, top_k=top_k)
        st.info("Sadece retrieval modu — cevap üretilmedi.")
        cevap_hits = hits
    else:
        try:
            with st.spinner("Getiriliyor ve cevap üretiliyor…"):
                t0 = time.perf_counter()
                r = answer(soru, top_k=top_k, model=chat_model)
                sure = time.perf_counter() - t0

            st.subheader("Cevap")
            if any(m in r["answer"].casefold() for m in RED_ISARETLERI):
                # Reddetme bir hata degil, tasarlanmis davranis -> oyle gorunsun.
                # Kaynak da gostermiyoruz: alakasiz chunk'lari kaynak diye sunmak yaniltir.
                st.info(r["answer"])
                st.caption("Bağlamda cevap bulunamadığı için sistem uydurmayı reddetti.")
            else:
                st.markdown(r["answer"])
                st.subheader("Kaynaklar")
                for s in r["sources"]:
                    st.markdown(f"- {s}")

            st.caption(f"{secim.split(' — ')[0]} · top-k={top_k} · {sure:.1f} sn")
            cevap_hits = r["hits"]
        except Exception as e:
            st.error(f"Cevap üretilemedi: {e}")
            st.markdown(
                "**Foundry çalışmıyor olabilir.** Terminalde:\n"
                "```bash\n"
                "foundry service status\n"
                "foundry service start\n"
                "foundry model load qwen2.5-1.5b\n"
                "export FOUNDRY_PORT=<status'taki port>\n"
                "```\n"
                "Bu arada kenar çubuğundan **Sadece retrieval** modunu açabilirsin."
            )
            cevap_hits = retrieve(soru, strategy=strategy, top_k=top_k)

    # Seffaflik paneli: RAG'in ic mekanigini goster
    with st.expander(f"Getirilen {len(cevap_hits)} chunk (benzerlik skoruyla)"):
        for i, h in enumerate(cevap_hits, 1):
            st.markdown(
                f"**{i}. {h['doc_name']} · s.{h['page']} · {h['section'] or '—'}** "
                f"— skor `{h['score']:.3f}`"
            )
            st.caption(" ".join(h["text"].split())[:400] + "…")
