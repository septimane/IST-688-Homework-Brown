import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF


def read_pdf(uploaded_file):
    """Read an uploaded PDF file and return its text."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# Show title and description.
st.title("My Document Question Answering")
st.write(
    "Upload a document below and ask a question about it. "
    "To use this app, you need an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

# Ask the user for their OpenAI API key.
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user pick which model to use.
    model = st.selectbox(
        "Which model would you like to use?",
        ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"],
    )

    # Let the user upload a file (only .txt and .pdf allowed).
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Figure out the file type and read the document accordingly.
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            document = None

        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API and stream it.
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            st.write_stream(stream)
