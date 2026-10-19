"""
One-time build script for HW 7.

Run from the repo root:
    python HW/build_hw7_db.py

Reads news.csv, embeds every article, and writes a ChromaDB collection to
HW/HW7_ChromaDB. The Streamlit app only opens that collection, so a user never
waits for 1,290 embeddings.
"""

import csv
import os
import sys
import tomllib

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "HW7-Data", "news.csv")
DB_DIR = os.path.join(HERE, "HW7_ChromaDB")
COLLECTION_NAME = "HW7Collection"
SECRETS = os.path.join(os.path.dirname(HERE), ".streamlit", "secrets.toml")

with open(SECRETS, "rb") as f:
    api_key = tomllib.load(f)["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Start clean so re-running the script does not duplicate articles.
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = chroma_client.create_collection(name=COLLECTION_NAME)

ids, documents, metadatas = [], [], []

with open(CSV_PATH, newline="", encoding="utf-8", errors="ignore") as f:
    for i, row in enumerate(csv.DictReader(f)):
        text = (row.get("Document") or "").strip()
        if not text:
            continue

        company = (row.get("company_name") or "Unknown").strip()
        date = (row.get("Date") or "")[:10]

        ids.append(f"article-{i}")
        # Company and date go into the embedded text so they are searchable,
        # not only filterable.
        documents.append(f"{company} ({date})\n{text}")
        metadatas.append({
            "company": company,
            "date": date,
            "url": (row.get("URL") or "").strip(),
        })

print(f"{len(documents)} articles to embed.")

BATCH = 100
for start in range(0, len(documents), BATCH):
    batch = documents[start:start + BATCH]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch,
    )
    collection.add(
        ids=ids[start:start + BATCH],
        documents=batch,
        embeddings=[d.embedding for d in response.data],
        metadatas=metadatas[start:start + BATCH],
    )
    print(f"  embedded {min(start + BATCH, len(documents))} of {len(documents)}")

print(f"Done. Collection has {collection.count()} articles at {DB_DIR}")