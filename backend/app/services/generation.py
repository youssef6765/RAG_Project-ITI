import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


PROMPT = ChatPromptTemplate.from_template(
    """
You are a document-grounded question answering assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information, say:
   "The retrieved document does not contain enough information to answer this question."
3. Do not invent facts, sources, or page numbers.
4. Cite the source page(s) used in this format:
   [Source: <filename>, p. <page>]
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
    def format_context(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            chunk = doc.metadata.get("chunk", "unknown")
            parts.append(
                f"[Context {i} | Source: {source} | Page: {page} | Chunk: {chunk}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def extract_sources(docs) -> list[str]:
        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            label = f"{source} p.{page}"
            if label not in seen:
                seen.add(label)
                sources.append(label)
        return sources

    def generate(self, question: str, docs):
        context = self.format_context(docs)
        response = self.chain.invoke(
            {"context": context, "question": question}
        )
        answer = response.content if hasattr(response, "content") else str(response)

        # If the model omitted citations, append the retrieved source list.
        # This keeps the API response source-grounded even when the local model
        # does not follow the citation instruction perfectly.
        if not re.search(r"\[Source:\s*.+?,\s*p\.\s*\d+\]", answer):
            source_text = ", ".join(self.extract_sources(docs))
            answer = f"{answer}\n\nSources: {source_text}"

        return answer, self.extract_sources(docs)
