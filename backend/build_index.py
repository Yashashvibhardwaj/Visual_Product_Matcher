import os
import json
import requests
import io
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # progress bar

# ---------------------------------------------------
# Locate products.json in the same folder as this script
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_FILE = os.path.join(BASE_DIR, "products.json")
INDEX_FILE = os.path.join(BASE_DIR, "products.index")

# ---------------------------------------------------
# Load product metadata
# ---------------------------------------------------
if not os.path.exists(PRODUCTS_FILE):
    raise FileNotFoundError(f"❌ Could not find {PRODUCTS_FILE}")

with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
    products = json.load(f)

print(f"📦 Loaded {len(products)} products from {PRODUCTS_FILE}")

# ---------------------------------------------------
# Load CLIP model
# ---------------------------------------------------
print("🧠 Loading CLIP model (this may take a few seconds)...")
model = SentenceTransformer("clip-ViT-B-32")

# ---------------------------------------------------
# Collect unique image URLs (avoid redundant downloads)
# ---------------------------------------------------
unique_urls = list({p["image_url"] for p in products})
print(f"🔗 Found {len(unique_urls)} unique image URLs")

# ---------------------------------------------------
# Compute embeddings for unique URLs
# ---------------------------------------------------
url_to_emb = {}

for url in tqdm(unique_urls, desc="Embedding unique images"):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        emb = model.encode(img, convert_to_numpy=True,
                           normalize_embeddings=True)
        url_to_emb[url] = emb
    except Exception as e:
        print(f"⚠️ Error processing {url}: {e}")
        url_to_emb[url] = np.zeros(512, dtype=np.float32)  # fallback embedding

# ---------------------------------------------------
# Build embeddings array for all products
# ---------------------------------------------------
embeddings = []
for p in products:
    embeddings.append(url_to_emb[p["image_url"]])

embeddings = np.array(embeddings).astype("float32")

print(f"✅ Built embeddings array: {embeddings.shape}")

# ---------------------------------------------------
# Create FAISS index (cosine similarity via inner product)
# ---------------------------------------------------
dim = embeddings.shape[1]  # 512 for CLIP
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# ---------------------------------------------------
# Save FAISS index
# ---------------------------------------------------
faiss.write_index(index, INDEX_FILE)
print(f"🎉 Saved FAISS index with {index.ntotal} vectors → {INDEX_FILE}")
