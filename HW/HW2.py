import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

st.title("🔗 HW 2 — URL Summarizer")


def read_url_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


# --- Sidebar options ---

provider = st.sidebar.selectbox("LLM provider", ["OpenAI", "Anthropic"])

use_advanced = st.sidebar.checkbox("Use advanced model")

MODELS = {
    "OpenAI": {False: "gpt-5-nano", True: "gpt-5-mini"},
    "Anthropic": {False: "claude-haiku-4-5-20251001", True: "claude-opus-5"},
}
model = MODELS[provider][use_advanced]
st.sidebar.write(f"Model in use: `{model}`")

summary_type = st.sidebar.selectbox(
    "Type of summary",
    [
        "Summarize the page in 100 words",
        "Summarize the page in 2 connecting paragraphs",
        "Summarize the page in 5 bullet points",
    ],
)

language = st.sidebar.selectbox(
    "Output language",
    ["English", "Spanish", "French", "German", "Chinese"],
)

# --- Main ---

url = st.text_input("Enter a web page URL")

if url:
    document = read_url_content(url)

    if not document:
        st.error("Could not read that URL. Check the address and try again.", icon="🚫")
        st.stop()

    prompt = (
        f"Here's the content of a web page:\n\n{document}\n\n---\n\n"
        f"{summary_type}. Write the summary in {language}."
    )

    try:
        if provider == "OpenAI":
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            st.write_stream(stream)
        else:
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            with client.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                st.write_stream(stream.text_stream)
    except Exception as e:
        st.error(f"{provider} call failed: {e}", icon="🚫")