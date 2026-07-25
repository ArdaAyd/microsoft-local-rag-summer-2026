"""PDF -> sayfa bazlı temiz metin (kaynak göstermek için sayfa bilgisi korunur)."""
import re
import pdfplumber


def _clean_line(line: str) -> str:
    return line.strip()


def _is_noise(line: str) -> bool:
    """Sayfa numarası, boş satır gibi işe yaramaz satırları at."""
    s = line.strip()
    if not s:
        return True
    # Tek başına sayfa numarası (örn '7', '23')
    if re.fullmatch(r"\d{1,3}", s):
        return True
    return False


def _serialize_tables(page) -> list[str]:
    """Tablo satırlarını metne ekler; extract_text()'in düşürdüğü hücre değerlerini kurtarır.

    'tablo: ' öneki küçük harf: chunker başlıkları büyük harfle tanıdığı için bu satırlar
    yanlışlıkla bölüm başlığı sanılmaz.
    """
    rows = []
    for table in page.extract_tables():
        for row in table:
            cells = [c for c in (" ".join((x or "").split()) for x in row) if c]
            if len(cells) >= 2:          # tek hücreli satır bilgi taşımaz
                rows.append("tablo: " + " | ".join(cells))
    return rows


def extract_pages(pdf_path: str) -> list[dict]:
    """PDF'i sayfa listesi olarak döner: [{'page': 1, 'text': '...'}, ...]"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            lines = [_clean_line(l) for l in raw.split("\n")]
            lines = [l for l in lines if not _is_noise(l)]
            lines += _serialize_tables(page)
            text = "\n".join(lines).strip()
            if text:
                pages.append({"page": i, "text": text})
    return pages


if __name__ == "__main__":
    # Hızlı test
    import sys
    from config import PDF_DIR, DOCS

    fname = DOCS[0][0] if len(sys.argv) < 2 else sys.argv[1]
    pages = extract_pages(str(PDF_DIR / fname))
    print(f"{fname}: {len(pages)} sayfa")
    print("--- Sayfa 7 önizleme ---")
    for p in pages:
        if p["page"] == 7:
            print(p["text"][:500])
