# pinecone_upload.py
import json
import time
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import config
import logging
import backoff
from openai import OpenAIError

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small
MAX_CHARS = 1000  # ✅ Added: truncate text safely

# -----------------------------
# Initialize logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# Initialize clients
# -----------------------------
try:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    raise

try:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize Pinecone client: {e}")
    raise

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    logging.info(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="gcp",
            region="us-east1-gcp"
        )
    )
else:
    logging.info(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
@backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)  # ✅ Added: retry logic
def get_embeddings(texts, model="text-embedding-3-small"):
    """Generate embeddings using OpenAI v1.0+ API with retry."""
    resp = client.embeddings.create(model=model, input=texts)
    return [data.embedding for data in resp.data]


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    
    for node in nodes:
        # ✅ Updated: safe text extraction and truncation
        semantic_text = (node.get("semantic_text") or node.get("description") or "").strip()[:MAX_CHARS]
        if not semantic_text:
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    logging.info(f"Preparing to upsert {len(items)} items to Pinecone...")

    total_batches = len(items) // BATCH_SIZE + (1 if len(items) % BATCH_SIZE != 0 else 0)
    for batch_num, batch in enumerate(tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"), 1):
        logging.info(f"Processing batch {batch_num}/{total_batches}")  # ✅ Added: progress tracking
        
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts, model="text-embedding-3-small")

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        # ✅ Added: error handling for Pinecone upsert
        try:
            index.upsert(vectors)
        except Exception as e:
            logging.error(f"Error uploading batch {batch_num}: {e}")
            continue

        time.sleep(0.2)

    logging.info("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()
