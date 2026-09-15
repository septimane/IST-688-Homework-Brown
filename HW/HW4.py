import os
import sys

# ChromaDB needs a newer sqlite3 than Streamlit Cloud ships with.
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import chromadb
from bs4 import BeautifulSoup
from openai import OpenAI

st.title("🍊 HW 4 — Syracuse Student Organization Chatbot")
st.write(
    "Ask about Syracuse student organizations. I search a vector database built from "
    "513 club pages and use the closest matches to answer."
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "HW4-Data")
DB_DIR = os.path.join(HERE, "HW4_ChromaDB")
COLLECTION_NAME = "HW4Collection"
MODEL = "gpt-5-mini"
BUFFER_TURNS = 5


def org_name(filename):
    """Turn the file name back into a readable organization name."""
    slug = filename.replace("syracuse.campuslabs.com_engage_organization_", "")
    slug = slug.replace(".html", "")
    return slug.replace("-", " ").replace("_", " ").strip().title()


def chunk_text(text):
    """
    CHUNKING METHOD: split each page into two chunks at the paragraph break
    nearest the midpoint.

    Why split at a paragraph break instead of the exact halfway character.
    Cutting at the raw midpoint regularly slices a sentence or a contact block
    in half. That produces a chunk whose embedding represents a fragment rather
    than an idea. Snapping to the closest blank line keeps both halves readable
    on their own, which is what the embedding needs to be meaningful.

    Why chunk at all. These club pages put different kinds of information in
    different places. The mission statement sits near the top. Contact details,
    meeting times and officer names sit near the bottom. A single embedding for
    a whole page averages those together and ends up vague. Two embeddings let a
    question about meeting times match the bottom half specifically, instead of
    competing with the mission statement for the same vector.
    """
    if len(text) < 400:
        return [text]

    midpoint = len(text) // 2
    breaks = [i for i in range(len(text)) if text.startswith("\n\n", i)]
    split_at = min(breaks, key=lambda i: abs(i - midpoint)) if breaks else midpoint

    return [text[:split_at].strip(), text[split_at:].strip()]


def build_vectordb():
    """Build the collection from the HTML pages. Persists to disk so it is created once."""
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # If the database already has content, skip the build entirely.
    if collection.count() > 0:
        return collection

    ids, documents, metadatas = [], [], []

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.lower().endswith(".html"):
            continue

        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not text:
            continue

        name = org_name(filename)

        for i, chunk in enumerate(chunk_text(text)):
            if not chunk:
                continue
            ids.append(f"{filename}::chunk{i}")
            # Org name goes into the embedded text so the club name itself is searchable.
            documents.append(f"{name}\n\n{chunk}")
            metadatas.append({"filename": filename, "organization": name, "chunk": i})

    # Embed in batches. One request per chunk would be roughly a thousand calls.
    progress = st.progress(0.0, text="Embedding club pages...")
    BATCH = 50

    for start in range(0, len(documents), BATCH):
        batch = documents[start:start + BATCH]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[d[:6000] for d in batch],
        )
        collection.add(
            ids=ids[start:start + BATCH],
            documents=batch,
            embeddings=[d.embedding for d in response.data],
            metadatas=metadatas[start:start + BATCH],
        )
        done = min(start + BATCH, len(documents))
        progress.progress(done / len(documents), text=f"Embedding {done} of {len(documents)} chunks...")

    progress.empty()
    return collection


if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Loading the vector database..."):
        st.session_state.HW4_VectorDB = build_vectordb()

st.caption(f"{st.session_state.HW4_VectorDB.count()} chunks indexed.")


def search(query, k=4):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return st.session_state.HW4_VectorDB.query(
        query_embeddings=[response.data[0].embedding],
        n_results=k,
    )


# --- Chat ---

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {"role": "assistant", "content": "Ask me about Syracuse student organizations."}
    ]

for msg in st.session_state.hw4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a club"):
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = search(prompt)
    documents = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"--- {m['organization']} ---\n{d[:3000]}"
        for m, d in zip(metas, documents)
    )

    system_prompt = (
        "You are a guide to Syracuse University student organizations.\n\n"
        "Below are the club pages most relevant to the question. Answer from them.\n"
        "Name the organizations you used at the start of your reply.\n"
        "If the answer is not in the material below, say so plainly and make clear "
        "you are answering from general knowledge instead.\n\n"
        f"RETRIEVED CLUB PAGES:\n{context}"
    )

    # Memory buffer: the last 5 interactions, which is 10 messages.
    buffer = st.session_state.hw4_messages[-(BUFFER_TURNS * 2):]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + buffer,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.hw4_messages.append({"role": "assistant", "content": response})