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

st.title("📚 RAG Document Assistant")
st.caption("Ask questions about the indexed document. Answers are grounded in retrieved passages.")

question = st.chat_input("Ask a question about the document...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                st.write(data["answer"])

                st.markdown("### Sources")
                for source in data.get("sources", []):
                    st.write(f"- {source}")

            except requests.exceptions.RequestException:
                st.error(
                    "The backend could not be reached. "
                    "Make sure FastAPI is running and API_BASE_URL is correct."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
