import json
import os
import sys

# ChromaDB needs a newer sqlite3 than Streamlit Cloud ships with.
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import chromadb
from openai import OpenAI

st.title("🔎 HW 5 — Smarter Club Chatbot")
st.write(
    "Same 513 Syracuse club pages as HW 4. The difference is that the model now "
    "decides when to search and writes its own search query, instead of the app "
    "searching the database on every single message."
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

HERE = os.path.dirname(__file__)
DB_DIR = os.path.join(HERE, "HW4_ChromaDB")
COLLECTION_NAME = "HW4Collection"
MODEL = "gpt-5-mini"
BUFFER_TURNS = 5


def get_collection():
    """Open the collection HW 4 already built. Nothing is rebuilt here."""
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


if "HW5_VectorDB" not in st.session_state:
    st.session_state.HW5_VectorDB = get_collection()

st.caption(f"{st.session_state.HW5_VectorDB.count()} chunks available from HW 4.")


def relevant_club_info(query, k=4):
    """
    The tool the LLM calls when it needs club information.

    Takes a search query written by the model, not the user's raw message, embeds
    it, and returns the closest matching club page chunks.
    """
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    results = st.session_state.HW5_VectorDB.query(
        query_embeddings=[response.data[0].embedding],
        n_results=k,
    )

    documents = results["documents"][0]
    metas = results["metadatas"][0]

    return [
        {"organization": m["organization"], "content": d[:3000]}
        for m, d in zip(metas, documents)
    ]


club_tool = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": (
            "Search Syracuse University student organization pages. Call this whenever "
            "the answer depends on club information, including activities, meeting "
            "times, contacts, officers or how to join."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A descriptive search query. Write a full phrase rather than "
                        "the user's raw words. If the user says 'what about their "
                        "meetings', write something like 'African Student Union "
                        "meeting times and location'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


SYSTEM_PROMPT = (
    "You are a guide to Syracuse University student organizations.\n"
    "Use the relevant_club_info tool whenever the answer depends on club information.\n"
    "Name the organizations you used at the start of your reply.\n"
    "If the search returns nothing useful, say so plainly and make clear you are "
    "answering from general knowledge instead."
)

if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [
        {"role": "assistant", "content": "Ask me about Syracuse student organizations."}
    ]

for msg in st.session_state.hw5_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a club"):
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Short term memory: the last 5 interactions.
    buffer = st.session_state.hw5_messages[-(BUFFER_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + buffer

    # First call. The model decides whether it needs to search, and what to search for.
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[club_tool],
        tool_choice="auto",
    )
    reply = first.choices[0].message

    if reply.tool_calls:
        call = reply.tool_calls[0]
        args = json.loads(call.function.arguments or "{}")
        search_query = args.get("query", prompt)

        results = relevant_club_info(search_query)

        with st.expander(f"Searched for: {search_query}"):
            st.write(", ".join(r["organization"] for r in results))

        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(results),
        })

        # Second call. No tools passed, so it must answer from what it got back.
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
            )
            response = st.write_stream(stream)
    else:
        with st.chat_message("assistant"):
            response = reply.content or ""
            st.markdown(response)

    st.session_state.hw5_messages.append({"role": "assistant", "content": response})