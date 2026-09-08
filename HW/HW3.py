import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

st.title("🔗 HW 3 — Chat About a URL")

st.write("""
This chatbot answers questions about up to two web pages you choose.

Paste one or two URLs in the sidebar. The text of those pages gets pulled in and
placed in a system prompt that is sent with every single message, so the bot
never loses the source material.

Conversation memory is a **buffer of 6 messages**, meaning the last 3 back and
forth exchanges. Anything older is dropped before the request goes out. The
system prompt holding the URL text sits outside that buffer, so it is never
trimmed away.

Pick which vendor's model answers using the sidebar.
""")


def read_url_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


# --- Sidebar ---

st.sidebar.header("Sources")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2 (optional)")

st.sidebar.header("Model")
vendor = st.sidebar.selectbox("Vendor", ["OpenAI", "Anthropic"])

MODELS = {"OpenAI": "gpt-6-astra", "Anthropic": "claude-fable-5-1"}
model = MODELS[vendor]
st.sidebar.write(f"Model in use: `{model}`")

BUFFER_SIZE = 6

# --- Load the pages ---

sources = []
for label, url in (("URL 1", url1), ("URL 2", url2)):
    if url:
        text = read_url_content(url)
        if text:
            sources.append(f"--- {label}: {url} ---\n{text}")
            st.sidebar.success(f"{label} loaded")
        else:
            st.sidebar.error(f"{label} could not be read")

if not sources:
    st.info("Add at least one URL in the sidebar to start.")
    st.stop()

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions using only the documents below. "
    "If the answer is not in them, say so plainly.\n\n"
    + "\n\n".join(sources)
)

# --- Chat ---

if "hw3_messages" not in st.session_state:
    st.session_state.hw3_messages = []

for msg in st.session_state.hw3_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the pages"):
    st.session_state.hw3_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    buffer = st.session_state.hw3_messages[-BUFFER_SIZE:]

    with st.chat_message("assistant"):
        try:
            if vendor == "OpenAI":
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + buffer,
                    stream=True,
                )
                response = st.write_stream(stream)
            else:
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                with client.messages.stream(
                    model=model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=buffer,
                ) as stream:
                    response = st.write_stream(stream.text_stream)
        except Exception as e:
            response = f"Something went wrong: {e}"
            st.error(response, icon="🚫")

    st.session_state.hw3_messages.append({"role": "assistant", "content": response})