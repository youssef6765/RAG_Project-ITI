import base64
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered",
)


def query_backend(question: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


st.title("📚 RAG Document Assistant")
st.caption("Ask questions about the indexed document. Answers are grounded in retrieved passages.")

question = st.chat_input("Ask a question about the document...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            try:
                data = query_backend(question)

                st.write(data["answer"])

                sources = data.get("sources", [])
                if sources:
                    st.markdown("### Sources")
                    for source in sources:
                        st.write(f"- {source}")

                images = data.get("images", [])
                if images:
                    st.markdown("### Retrieved Images")
                    for img_b64 in images:
                        try:
                            st.image(base64.b64decode(img_b64))
                        except Exception:
                            pass  # skip anything that isn't valid image data

            except requests.exceptions.RequestException:
                st.error(
                    "The backend could not be reached. "
                    "Make sure FastAPI is running and API_BASE_URL is correct."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")