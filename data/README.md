# `data/` — source documents

The four source PDFs are the **official Teknofest 2026 Robotaxi competition documents**. They are
not redistributed in this repository; download them from the official Teknofest / T3 Foundation
sources and place them here with these exact filenames (matched in `src/config.py` → `DOCS`):

| Filename | Short name |
|---|---|
| `2026_Robotaksi-Şartnamesi.pdf` | Sartname |
| `2026_Araç_Kullanıcı_Dokümanı.pdf` | KullaniciDokumani |
| `2026_Araç_Genel_Bilgilendirme.pdf` | GenelBilgilendirme |
| `2026_Araç-Mimari.pdf` | Mimari |

Once the PDFs are in place, build the vector database:

```bash
cd src && python ingest.py
```

This creates `data/corpus.db` (also git-ignored — it is a generated artifact, ~9 MB).
