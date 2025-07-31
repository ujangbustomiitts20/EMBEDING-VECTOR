import os
import json
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
# === KONFIGURASI ===
OCR_JSON_DIR = "./perwal/json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "versi2"
MODEL_NAME = "intfloat/multilingual-e5-base"
LOG_FILE = "progress_log.json"
BATCH_SIZE = 32
# === LOAD MODEL EMBEDDING ===
print(f"🔤 Memuat model embedding: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
# === INISIALISASI QDRANT ===
client = QdrantClient(url=QDRANT_URL)
if not client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"📁 Koleksi '{COLLECTION_NAME}' belum ada. Membuat koleksi baru...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
else:
    print(f"📚 Menggunakan koleksi Qdrant yang sudah ada: {COLLECTION_NAME}")
# === PROGRESS LOG ===
progress_log = {"success": [], "failed": []}
# === EMBEDDING DAN INDEXING ===
def index_json_chunks(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    print(f"📦 Menemukan {len(files)} file JSON untuk diproses...\n")
    for file in tqdm(files, desc="🚀 Memproses JSON", unit="file"):
        file_path = os.path.join(folder, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"❌ Gagal membaca file {file}: {e}")
            progress_log["failed"].append({"file": file, "reason": str(e)})
            continue
        documents, metadatas, ids = [], [], []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            documents.append(text)
            ids.append(str(uuid.uuid4()))
            metadatas.append({
                "filename": chunk.get("filename", file),
                "chunk_index": i,
                "chunk_id": chunk.get("chunk_id", f"{file}_chunk_{i}")
            })
        if not documents:
            print(f"⚠️  File {file} tidak memiliki teks yang valid.")
            progress_log["failed"].append({"file": file, "reason": "No valid text"})
            continue
        total_success = 0
        try:
            for i in range(0, len(documents), BATCH_SIZE):
                batch_docs = documents[i:i+BATCH_SIZE]
                batch_ids = ids[i:i+BATCH_SIZE]
                batch_metas = metadatas[i:i+BATCH_SIZE]
                try:
                    batch_vectors = model.encode(batch_docs, show_progress_bar=False).tolist()
                except Exception as embed_err:
                    print(f"❌ Gagal embedding batch {i // BATCH_SIZE} di file {file}: {embed_err}")
                    continue
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        wait=True,
                        points=[
                            PointStruct(id=batch_ids[j], vector=batch_vectors[j],
                                        payload={**batch_metas[j], "text": batch_docs[j]})
                            for j in range(len(batch_docs))
                        ]
                    )
                    total_success += len(batch_docs)
                except Exception as upsert_err:
                    print(f"❌ Gagal upsert ke Qdrant (batch {i // BATCH_SIZE}) di file {file}: {upsert_err}")
                    continue
            if total_success > 0:
                print(f" {file}: {total_success} chunk berhasil diindex.")
                progress_log["success"].append({"file": file, "chunks": total_success})
            else:
                progress_log["failed"].append({"file": file, "reason": "All batches failed"})
        except Exception as e:
            print(f"❌ Gagal proses seluruh file {file}: {e}")
            progress_log["failed"].append({"file": file, "reason": str(e)})
# === EKSEKUSI ===
if __name__ == "__main__":
    print("🚀 Mulai indexing ke Qdrant...")
    index_json_chunks(OCR_JSON_DIR)
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        json.dump(progress_log, logf, indent=2, ensure_ascii=False)
    print(f"\n📝 Progress disimpan di '{LOG_FILE}'")
    print(" Proses indexing selesai.")
