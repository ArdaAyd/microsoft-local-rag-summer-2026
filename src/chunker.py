"""Chunking stratejileri. İkisi de aynı imzayı taşır:
    strategy(pages: list[dict], doc_name: str) -> list[dict]
Dönen her chunk: {doc_name, page, section, text, strategy}

  - fixed:   sabit karakter boyutu + overlap (baseline)
  - heading: başlık/madde bazlı (Teknofest dokümanları maddeli)
İkisi de aynı DB'ye 'strategy' etiketiyle yazılır; karşılaştırma bkz. docs/EXPERIMENTS.md.
"""
import re


# --- Strateji 1: sabit boyut + overlap ---
def chunk_fixed(pages, doc_name, size=800, overlap=150):
    # Tüm sayfaları birleştir ama karakter -> sayfa eşlemesini koru
    full, char_page = [], []
    for p in pages:
        for ch in p["text"] + "\n":
            full.append(ch)
            char_page.append(p["page"])
    full = "".join(full)

    chunks = []
    start = 0
    while start < len(full):
        end = min(start + size, len(full))
        text = full[start:end].strip()
        if text:
            page = char_page[start] if start < len(char_page) else char_page[-1]
            chunks.append({
                "doc_name": doc_name,
                "page": page,
                "section": "",              # sabit stratejide bölüm bilgisi yok
                "text": text,
                "strategy": "fixed",
            })
        start += size - overlap
    return chunks


# --- Strateji 2: başlık/madde bazlı ---
# Teknofest dokümanlarında başlıklar tipik olarak kısa, büyük harfle başlayan,
# noktalama içermeyen satırlar. Basit bir sezgi ile bölüm sınırı buluyoruz.
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-ZÇĞİÖŞÜ][^\.:]{2,60}$")


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or len(s) > 60:
        return False
    if s.endswith((".", ",", ";", ":")):
        return False
    # Başlıkların çoğu harf ağırlıklıdır (JSON/koordinat satırlarını eleyelim)
    letters = sum(c.isalpha() for c in s)
    if letters < len(s) * 0.5:
        return False
    return bool(_HEADING_RE.match(s))


def chunk_heading(pages, doc_name, max_chars=1200):
    chunks = []
    cur_section = "Giris"
    cur_lines, cur_page = [], pages[0]["page"] if pages else 1

    def flush():
        nonlocal cur_lines
        text = "\n".join(cur_lines).strip()
        if text:
            chunks.append({
                "doc_name": doc_name,
                "page": cur_page,
                "section": cur_section,
                "text": text,
                "strategy": "heading",
            })
        cur_lines = []

    for p in pages:
        for line in p["text"].split("\n"):
            if _looks_like_heading(line):
                flush()
                cur_section = line.strip()
                cur_page = p["page"]
            else:
                cur_lines.append(line)
                # Çok uzarsa böl (tek bölüm devasa olmasın)
                if sum(len(x) for x in cur_lines) > max_chars:
                    flush()
                    cur_page = p["page"]
    flush()
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "heading": chunk_heading}


if __name__ == "__main__":
    from config import PDF_DIR, DOCS
    from pdf_extract import extract_pages

    fname, short = DOCS[0]
    pages = extract_pages(str(PDF_DIR / fname))
    for name, fn in STRATEGIES.items():
        cs = fn(pages, short)
        print(f"{name}: {len(cs)} chunk | örnek section: {cs[len(cs)//2]['section']!r}")
        print("   ", cs[len(cs)//2]["text"][:120].replace("\n", " "), "...")
