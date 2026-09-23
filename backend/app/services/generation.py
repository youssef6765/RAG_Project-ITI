import re
from base64 import b64decode

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


PROMPT = ChatPromptTemplate.from_template(
    """
You are a document-grounded question answering assistant.

Answer the user's question using ONLY the retrieved context below, which may
include text, tables, and images.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information, say:
   "The retrieved document does not contain enough information to answer this question."
3. Do not invent facts, sources, or page numbers.
4. Cite the source page(s) used in this format:
   [Source: p. <page>]
5. Keep the answer concise but complete.

Retrieved context:
{context}

User question:
{question}

Answer:
"""
)


class GenerationService:
    def __init__(self, model_name: str, temperature: float = 0):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.chain = PROMPT | self.llm

    @staticmethod
    def _is_base64_image(doc) -> bool:
        if not isinstance(doc, str):
            return False
        try:
            b64decode(doc, validate=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _split_docs(docs):
        """Separate raw docstore items into text/table elements vs base64 images."""
        text_like, images = [], []
        for doc in docs:
            if GenerationService._is_base64_image(doc):
                images.append(doc)
            else:
                text_like.append(doc)
        return text_like, images

    @staticmethod
    def format_context(text_like) -> str:
        parts = []
        for i, doc in enumerate(text_like, start=1):
            text = getattr(doc, "text", None) or str(doc)
            page = getattr(doc.metadata, "page_number", "unknown") if hasattr(doc, "metadata") else "unknown"
            parts.append(f"[Context {i} | Page: {page}]\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def extract_sources(text_like) -> list[str]:
        sources = []
        seen = set()
        for doc in text_like:
            page = getattr(doc.metadata, "page_number", "unknown") if hasattr(doc, "metadata") else "unknown"
            label = f"p.{page}"
            if label not in seen:
                seen.add(label)
                sources.append(label)
        return sources

    def generate(self, question: str, docs):
        text_like, images = self._split_docs(docs)
        context = self.format_context(text_like)

        response = self.chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)

        if not re.search(r"\[Source:\s*p\.\s*\d+\]", answer):
            source_text = ", ".join(self.extract_sources(text_like)) or "none"
            answer = f"{answer}\n\nSources: {source_text}"

        return answer, self.extract_sources(text_like), images