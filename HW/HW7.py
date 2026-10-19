import json
import os
import sys

# ChromaDB needs a newer sqlite3 than Streamlit Cloud ships with.
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import chromadb
from openai import OpenAI

st.title("📰 HW 7 — Client News Monitor")
st.write(
    "A news bot for a law firm tracking client coverage. It answers only from a fixed "
    "set of 1,290 articles covering 177 companies between July 30 and August 6, 2024."
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

HERE = os.path.dirname(__file__)
DB_DIR = os.path.join(HERE, "HW7_ChromaDB")
COLLECTION_NAME = "HW7Collection"
BUFFER_TURNS = 5

# Requirement 1b. "Interesting" is a judgment, not a search term, so it needs an
# explicit definition. A law firm monitoring clients cares about legal and
# commercial exposure, not product launches.
INTERESTING_CRITERIA = """A story is "interesting" to this law firm when it signals legal
or commercial exposure for a client. In rough priority order:

1. Litigation and regulatory action. Lawsuits, investigations, antitrust, fines, subpoenas.
2. Deals. Mergers, acquisitions, divestitures, major investments.
3. Governance shocks. Executive departures, board changes, restructuring, layoffs.
4. Security and compliance incidents. Breaches, data loss, policy violations.

Product launches, earnings beats and routine partnerships rank low unless they carry
one of the above."""

# The standing query used to pull candidates for ranking. Vector search cannot judge
# importance, so this describes what importance looks like in article language.
INTERESTING_QUERY = (
    "lawsuit litigation legal action court filing settlement regulatory investigation "
    "antitrust probe subpoena fine penalty data breach security incident class action "
    "merger acquisition takeover divestiture executive resignation CEO departure "
    "layoffs restructuring bankruptcy fraud allegation compliance violation"
)

MODELS = {
    "Lower cost (gpt-5-nano)": "gpt-5-nano",
    "Higher cost (gpt-6-astra)": "gpt-6-astra",
}

model_label = st.sidebar.selectbox("Model", list(MODELS.keys()))
model = MODELS[model_label]
st.sidebar.caption(f"Using `{model}`")


@st.cache_resource
def get_collection():
    """Open the collection built by build_hw7_db.py. Nothing is embedded at runtime."""
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


collection = get_collection()
st.caption(f"{collection.count()} articles loaded.")


def _search(query, limit):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    results = collection.query(
        query_embeddings=[response.data[0].embedding],
        n_results=limit,
    )
    return [
        {"company": m["company"], "date": m["date"], "url": m["url"], "article": d}
        for m, d in zip(results["metadatas"][0], results["documents"][0])
    ]


def find_news(query, limit=8):
    """Tool: articles about a specific company or topic."""
    return _search(query, limit)


def rank_interesting_news(limit=20):
    """
    Tool: candidates for the "most interesting" ranking.

    Two stage by design. Vector search retrieves against the standing
    interestingness query, then the model ranks what comes back. Embeddings can
    find legally-flavoured language but cannot weigh one story against another.
    """
    return _search(INTERESTING_QUERY, limit)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_news",
            "description": (
                "Search the news set for articles about a specific company, industry "
                "or topic. Use for any question naming a company or subject."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive search phrase, for example 'Intel antitrust investigation'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many articles to return. Default 8.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_interesting_news",
            "description": (
                "Get candidate articles for a ranked list of the most important news. "
                "Use when the user asks what is interesting, important, notable or "
                "worth attention, without naming a company."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many candidates to pull. Default 20.",
                    }
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You monitor news for a global law firm that tracks coverage of its
clients. You answer only from the articles returned by your tools. Never add news from
your own knowledge.

{INTERESTING_CRITERIA}

Rules for your answers:
- When asked what is interesting or important, return a numbered ranked list. For each
  item give the company, the headline, and one sentence saying why it ranks there,
  referring to the criteria above.
- When asked about a specific company or topic, report what the articles say and name
  the companies and dates.
- Always include the article URL for anything you cite.
- If the tools return nothing relevant, say so plainly. Do not fill the gap.
- The set covers July 30 to August 6, 2024 only. Say so if asked about anything outside that."""

if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = [
        {"role": "assistant", "content": "Ask me what is interesting this week, or about a specific client."}
    ]

for msg in st.session_state.hw7_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the news"):
    st.session_state.hw7_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    buffer = st.session_state.hw7_messages[-(BUFFER_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + buffer

    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    reply = first.choices[0].message

    if reply.tool_calls:
        call = reply.tool_calls[0]
        args = json.loads(call.function.arguments or "{}")

        if call.function.name == "rank_interesting_news":
            results = rank_interesting_news(args.get("limit", 20))
            label = f"Pulled {len(results)} candidates for ranking"
        else:
            results = find_news(args.get("query", prompt), args.get("limit", 8))
            label = f"Searched for: {args.get('query', prompt)}"

        with st.expander(label):
            st.write(", ".join(sorted({r["company"] for r in results})))

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

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            response = st.write_stream(stream)
    else:
        with st.chat_message("assistant"):
            response = reply.content or ""
            st.markdown(response)

    st.session_state.hw7_messages.append({"role": "assistant", "content": response})