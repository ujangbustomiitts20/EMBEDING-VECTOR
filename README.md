#  JSON Embedding & Indexing to Qdrant

Skrip ini digunakan untuk melakukan **embedding teks** dari file `.json` dan mengindeksnya ke dalam vektor database **Qdrant** menggunakan model `sentence-transformers`. Sangat cocok digunakan dalam sistem berbasis pencarian semantik, RAG (Retrieval-Augmented Generation), atau asisten dokumen cerdas.

## Fitur

- Membaca semua file `.json` berisi potongan teks (chunk) dari folder tertentu
- Melakukan embedding menggunakan model `intfloat/multilingual-e5-base`
- Indexing ke **Qdrant Vector Database** dengan payload metadata
- Proses dilakukan secara batch untuk efisiensi
- Logging proses sukses dan gagal ke file `progress_log.json`

## Struktur Folder

```
project_root/
├── index_json_qdrant.py     # Skrip utama indexing
├── perwal/
│   └── json/                # Folder input berisi file-file .json hasil chunking teks
├── progress_log.json        # File log hasil proses (otomatis dibuat)
```

## Persiapan

1. **Install dependensi**
```bash
pip install sentence-transformers qdrant-client tqdm
```

2. **Pastikan Qdrant sudah berjalan secara lokal**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## ⚙️ Konfigurasi dalam Skrip

```python
OCR_JSON_DIR = "./perwal/json"              # Folder input berisi JSON
QDRANT_URL = "http://localhost:6333"        # Alamat Qdrant
COLLECTION_NAME = "versi2"                  # Nama koleksi Qdrant
MODEL_NAME = "intfloat/multilingual-e5-base"# Model embedding
LOG_FILE = "progress_log.json"              # Lokasi file log
BATCH_SIZE = 32                             # Ukuran batch untuk embedding
```

## Cara Menjalankan

1. Pastikan folder `perwal/json/` berisi file JSON hasil chunking (struktur setiap elemen harus memiliki kunci `text`)
2. Jalankan skrip:

```bash
python index_json_qdrant.py
```

3. Jika berhasil, terminal akan menampilkan jumlah chunk yang berhasil diindex. Semua log (berhasil/gagal) akan ditulis ke `progress_log.json`.

## Contoh Format Input JSON

Setiap file JSON berisi daftar seperti ini:

```json
[
  {
    "filename": "dokumen1.pdf",
    "chunk_id": "dokumen1.pdf_chunk_0",
    "text": "Ini adalah isi teks chunk pertama."
  },
  ...
]
```

## Catatan

- Jika koleksi Qdrant belum ada, skrip akan membuatnya secara otomatis.
- Data yang di-index mencakup: vektor teks, metadata (`filename`, `chunk_id`, dll), dan teks asli.
- Gunakan log `progress_log.json` untuk analisis ulang bila ada kegagalan.

---

Dibangun menggunakan [Qdrant](https://qdrant.tech), [Sentence-Transformers](https://www.sbert.net), dan Python.  
