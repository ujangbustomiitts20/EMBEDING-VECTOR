

### 📄 README (English)

```markdown
# EMBEDING-VECTOR

A Python script for embedding text chunks from JSON files and indexing them into a Qdrant vector database using a multilingual embedding model. Ideal for semantic search systems, RAG (Retrieval-Augmented Generation), or intelligent document assistants.

## Features  
- Read all `.json` files containing text chunks from a specified folder.  
- Embed chunks using the model `intfloat/multilingual-e5-base` (via Sentence‑Transformers).  
- Index embeddings into a Qdrant vector database with metadata payload (e.g., filename, chunk_id).  
- Batch processing for efficiency.  
- Write a log file (`progress_log.json`) for success/failure tracking.

## Folder Structure  
```

project_root/
├── index_json_qdrant.py       # Main indexing script
├── perwal/
│   └── json/                  # Input folder containing .json files from text chunking
└── progress_log.json          # Auto-generated log file

````

## Setup  
1. Install dependencies:  
   ```bash
   pip install sentence-transformers qdrant-client tqdm
````

2. Ensure Qdrant is running locally:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Configuration (in the script)

```python
OCR_JSON_DIR = "./perwal/json"                    # Folder of input JSON  
QDRANT_URL = "http://localhost:6333"               # Qdrant endpoint  
COLLECTION_NAME = "versi2"                          # Qdrant collection name  
MODEL_NAME = "intfloat/multilingual-e5-base"       # Embedding model  
LOG_FILE = "progress_log.json"                      # Log file path  
BATCH_SIZE = 32                                     # Batch size for embedding  
```

## Usage

1. Place your chunked JSON files in `perwal/json/` — each chunked JSON should contain items like:

   ```json
   [
     {
       "filename": "document1.pdf",
       "chunk_id": "document1.pdf_chunk_0",
       "text": "This is the text of the first chunk."
     },
     …
   ]
   ```
2. Run:

   ```bash
   python index_json_qdrant.py
   ```
3. After execution, the terminal will display how many chunks were indexed. Use `progress_log.json` to check success or error details.

## Notes

* If the specified Qdrant collection does not exist, the script will automatically create it.
* Indexed data includes: the text vector, metadata (`filename`, `chunk_id`, etc.), and the original text.
* Use `progress_log.json` to analyze failures or monitor process status.

---

Built with Qdrant, Sentence‑Transformers and Python.

---

## License

This project is licensed under the GPL-3.0 License.

```

---
