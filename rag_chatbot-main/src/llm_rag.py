import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from langchain_core.documents import Document
from vector_store import VectorStore

DEFAULT_GEMINI_API_KEY = "" #вставить апи
# DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

class LLMRAGHandler:
    """
    A class to handle LLM-based RAG (Retrieval-Augmented Generation) tasks.
    
    Attributes:
        llm (ChatOllama): The language model used for generating responses.
        vector_store (VectorStore): The vector store used for document retrieval.
        system_prompt (str): The system prompt given to the model.
        history (List[BaseMessage]): The conversation history.
        rag_prompt (PromptTemplate): The prompt template for q&a with RAG.
        llm_chain (Chain): The chain for RAG.
        rag_chain (Chain): The retrieval chain.
    
    Methods:
        __init__(self, model="granite3.3"): Initializes the LLMRAGHandler with the specified model.
        generate_response(self, human_message) -> AIMessage: Generates and appends a response from the LLM.
        reset(self) -> None: Resets the conversation history.
        get_history(self) -> List[BaseMessage]: Returns the conversation history.
        retrieve(self, question: str, k:int = 4) -> List[Document]: Retrieves the most relevant documents for a given question.
        add_pdf_to_context(self, filePath: Path): Adds a PDF file to the context for retrieval.

    """
    def __init__(self, model="gemma:2b", gemini_api_key: str | None = None):
        """
        Initializes the LLMRAGHandler with the specified model.

        Args:
            model (str): The model to use for the language model and vector store. Default is "gemma:2b".
            gemini_api_key (str | None): The Gemini API key for generation. If not provided, it is read from GEMINI_API_KEY env var.
        """
        self.model = model
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)
        self.gemini_endpoint = os.environ.get("GEMINI_ENDPOINT", DEFAULT_GEMINI_ENDPOINT)
        self.vector_store = VectorStore(llm_model=model)
        
        # System prompt - These are the instructions for the model
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you cite text, mention which document, page, and line range you used. "
            "If the context does not provide enough information, say so clearly. "
            "Keep the answer concise."
        )
        
        self.history: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]

    def _format_documents(self, docs: List[Document]) -> str:
        if not docs:
            return ""

        formatted = []
        for idx, doc in enumerate(docs, start=1):
            ref = doc.metadata.get("source_ref") or doc.metadata.get("source") or "unknown source"
            text = (doc.page_content or "").strip()
            if len(text) > 1200:
                text = text[:1200].rstrip() + "..."
            formatted.append(f"Source {idx}: {ref}\n{text}")
        return "\n\n".join(formatted)

    def _build_prompt(self, question: str, docs: List[Document]) -> str:
        context_text = self._format_documents(docs)
        if context_text:
            return (
                f"{self.system_prompt}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                "Answer the question using only the provided context. "
                "If the answer cannot be found in the context, say that it is not available."
            )
        return f"{self.system_prompt}\n\nQuestion: {question}\nAnswer:"

    def _build_source_listing(self, docs: List[Document]) -> str:
        source_refs = []
        for doc in docs:
            ref = doc.metadata.get("source_ref") or doc.metadata.get("source")
            snippet = doc.page_content.strip().replace("\n", " ")
            snippet = snippet[:240].rstrip()
            if ref and ref not in source_refs:
                source_refs.append(ref)
        if not source_refs:
            return ""
        output = [f"{idx + 1}. {ref}" for idx, ref in enumerate(source_refs)]
        return "\n".join(output)

    def _extract_gemini_text(self, response_json: dict) -> str:
        candidates = response_json.get("candidates") or []
        if candidates:
            first = candidates[0]
            content = first.get("content") or {}
            parts = content.get("parts") or []
            if isinstance(parts, list) and parts:
                texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
                return "\n".join(texts).strip()
        return response_json.get("text", "") or ""

    def _call_gemini(self, prompt: str) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "Gemini API key is not configured. Set GEMINI_API_KEY environment variable or pass gemini_api_key to LLMRAGHandler."
            )

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.gemini_api_key,
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ],
                }
            ],
        }

        timeout = int(os.environ.get("GEMINI_TIMEOUT", "60"))
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            backoff_factor=1,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        try:
            response = session.post(
                self.gemini_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Gemini API request timed out after {timeout} seconds. "
                f"Увеличьте GEMINI_TIMEOUT или проверьте интернет-соединение. "
                f"Ошибка: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Ошибка при запросе в Gemini API: {exc}"
            ) from exc

        return self._extract_gemini_text(response.json())

    def generate_response(self, human_message) -> str:
        """
        Generates and appends a response from the Gemini API.

        Args:
            human_message (str): The user's message.

        Returns:
            str: The AI's response with citations appended.
        """
        print("Generating response from Gemini API...")
        context_docs = self.retrieve(human_message)
        prompt = self._build_prompt(human_message, context_docs)
        answer = self._call_gemini(prompt)

        citation_text = self._build_source_listing(context_docs)
        if citation_text:
            answer = f"{answer}\n\nИсточники:\n{citation_text}"

        self.history.append(HumanMessage(content=human_message))
        self.history.append(AIMessage(content=answer))
        return answer

    def reset(self) -> None:
        """
        Resets the conversation history.
        """
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

    def get_history(self) -> List[BaseMessage]:
        """
        Returns the conversation history.

        Returns:
            List[BaseMessage]: The conversation history.
        """       
        return self.history
    
    def retrieve(self, question: str, k:int = 4) -> List[Document]:
        """
        Retrieves the most relevant documents for a given question.

        Args:
            question (str): The question to retrieve documents for.
            k (int): The number of documents to retrieve. Default is 4.

        Returns:
            List[Document]: The retrieved documents.
        """
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        return retrieved_docs

    
    def add_pdf_to_context(self, filePath: Path) -> List[Document]:
        """
        Adds a PDF file to the context for retrieval.

        Args:
            filePath (Path): The path to the PDF file.
        Returns:
            List[Document]: The documents added to the vector store.
        """
        self.vector_store.add_document(filePath)
    
if __name__ == '__main__':
    print("Run this module through chat_ui.py or import LLMRAGHandler.")