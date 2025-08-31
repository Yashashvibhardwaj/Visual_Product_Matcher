from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import io
import faiss
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# Init FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model + catalog
model = SentenceTransformer("clip-ViT-B-32")

with open("products.json", "r", encoding="utf-8", errors="ignore") as f:
    products = json.load(f)


# Load FAISS index
index = faiss.read_index("products.index")


def embed_image(img: Image.Image):
    return model.encode(img, convert_to_numpy=True, normalize_embeddings=True)


def embed_text(query: str):
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


@app.post("/match")
async def match(
    file: UploadFile = None,
    image_url: str = Form(None),
    min_score: float = Form(0.6),
    top_k: int = Form(60),
    categories: str = Form(None),
    brands: str = Form(None),
    min_price: float = Form(0),
    max_price: float = Form(9999)
):
    try:
        # Get query image
        if file:
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        elif image_url:
            img = Image.open(io.BytesIO(requests.get(
                image_url).content)).convert("RGB")
        else:
            return {"matches": []}

        # Encode query
        q_emb = embed_image(img).reshape(1, -1)

        # Search FAISS
        scores, ids = index.search(q_emb, top_k)

        # Parse filters
        categories = json.loads(categories) if categories else []
        brands = json.loads(brands) if brands else []

        # Collect results
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if score < min_score:
                continue
            p = products[idx]

            # Apply filters
            if categories and p["category"] not in categories:
                continue
            if brands and p["brand"] not in brands:
                continue
            if not (min_price <= p["price"] <= max_price):
                continue

            results.append({**p, "score": float(score)})
        return {"matches": results}
    except Exception as e:
        return {"error": str(e)}


@app.post("/search_text")
async def search_text(
    query: str = Form(...),
    min_score: float = Form(0.6),
    top_k: int = Form(60),
    categories: str = Form(None),
    brands: str = Form(None),
    min_price: float = Form(0),
    max_price: float = Form(9999)
):
    try:
        # Encode text query
        q_emb = embed_text(query).reshape(1, -1)

        # Search FAISS
        scores, ids = index.search(q_emb, top_k)

        # Parse filters
        categories = json.loads(categories) if categories else []
        brands = json.loads(brands) if brands else []

        # Collect results
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if score < min_score:
                continue
            p = products[idx]

            # Apply filters
            if categories and p["category"] not in categories:
                continue
            if brands and p["brand"] not in brands:
                continue
            if not (min_price <= p["price"] <= max_price):
                continue

            results.append({**p, "score": float(score)})
        return {"matches": results}
    except Exception as e:
        return {"error": str(e)}
