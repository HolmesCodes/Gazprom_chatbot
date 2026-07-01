import logging
import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
import faiss
import bs4

# Constants
model = "gemma:2b"
llm = ChatOllama(model=model)
VECTOR_STORE_PATH = Path("faiss_index")


class VectorStore:
    """
    A class for managing a vector store of documents.

    Attributes:
        vector_store_path (str): The path to the vector store.
        llm_model (str): The language model used for embeddings.
        embeddings_model (OllamaEmbeddings): The embeddings model.
        chunk_size (int): The size of chunks for document splitting.
        chunk_overlap (int): The overlap between chunks for document splitting.
        persist (bool): Whether to persist the vector store to disk.
        index_path (str): The path to save the index.

    Methods:
        __init__(vector_store_path=VECTOR_STORE_PATH, llm_model="granite3.3",
                  chunk_size=500, chunk_overlap=50, persist=True, index_path="faiss_index"):
            Initializes the VectorStore object.

        _setup_vector_store():
            Sets up the vector store, either loading from disk or creating a new one.

        load_documents(data_path) -> List[Document]:
            Loads documents from the specified directory.

        add_documents(documents: List[Document]) -> List[Document]:
            Adds documents to the vector store and saves the index.

        chunk_documents(documents: List[Document]) -> List[Document]:
            Splits documents into smaller chunks.

        add_all_documents(data_path: str = "data") -> List[Document]:
            Loads and adds all documents from the specified directory.

        load_document(pdf_path: Path) -> List[Document]:
            Loads a single document from a PDF file.

        add_document(filePath: Path) -> List[Document]:
            Adds a single document from a PDF file.

        similarity_search(question: str, k:int) -> List[Document]:
            Performs a similarity search based on the given question.

        as_retriever() -> VectorStoreRetriever:
            Returns a retriever object for the vector store.

        index_websites(urls: list[str]) -> List[Document]:
            Indexes documents from the given URLs.

        website_to_documents(urls: list[str]) -> list[Document]:
            Converts URLs to LangChain Document objects with metadata.
    """
    def __init__(self, vector_store_path=VECTOR_STORE_PATH, llm_model="gemma:2b",
                  chunk_size=500, chunk_overlap=50, persist=True, index_path="faiss_index"):
        """
        Initialize the VectorStore class.

        Parameters:
        vector_store_path (str): The path to the directory where the vector store will be saved.
        llm_model (str): The name of the language model to use for generating embeddings.
        chunk_size (int): The number of characters per document chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
        persist (bool): Whether to persist the vector store to disk.
        index_path (str): The directory path to save the Faiss index.
        """
        self.vector_store_path = Path(vector_store_path)
        self.llm_model = llm_model
        self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist = persist
        self.index_path = Path(index_path)

        self._setup_vector_store()

    def _setup_vector_store(self) -> None:
        """
        Sets up the vector store for the model.

        This method loads an existing vector store from disk if it exists.
        If no persisted store is available, it creates a new FAISS store with
        the requested embeddings model.
        """
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        if self.vector_store_path.exists() and any(self.vector_store_path.iterdir()):
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            return

        self.embedding_dim = len(self.embeddings_model.embed_query("hello world"))
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings_model,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        if self.persist:
            self.vector_store.save_local(self.index_path)

    def load_documents(self, data_path) -> List[Document]:
        """
        Loads documents from the specified directory.

        Args:
            data_path (str): The path to the directory containing PDF files.

        Returns:
            List[str]: A list of document texts, each representing a document loaded from a PDF file.
        """
        documents = []
        for pdf_path in Path(data_path).glob("*.pdf"):
            docs = self.load_document(pdf_path)
            print(len(docs))
            documents.extend(docs)
        return documents
    
    def add_documents(self, documents: List[Document]) -> List[Document]:
        """
        Adds a list of documents to the vector store after splitting them into chunks.

        Args:
            documents (List[Document]): The list of documents to be added.

        Returns:
            List[Document]: The list of document chunks that were added to the vector store.
        """
        splitted_docs = self.chunk_documents(documents=documents)
        if not splitted_docs:
            return []

        texts = [doc.page_content for doc in splitted_docs]
        metadatas = [doc.metadata for doc in splitted_docs]
        ids = [doc.id or str(uuid.uuid4()) for doc in splitted_docs]

        embeddings = self.embeddings_model.embed_documents(texts)
        if embeddings is None:
            raise RuntimeError("Failed to compute embeddings for document chunks.")

        if isinstance(embeddings, list) and embeddings and not isinstance(
            embeddings[0], (list, tuple, np.ndarray)
        ):
            embeddings = [embeddings]

        vector = np.array(embeddings, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if getattr(self.vector_store, "_normalize_L2", False):
            faiss.normalize_L2(vector)

        self.vector_store.index.add(vector)
        self.vector_store.docstore.add(
            {
                id_: Document(id=id_, page_content=text, metadata=meta)
                for id_, text, meta in zip(ids, texts, metadatas)
            }
        )

        starting_len = len(self.vector_store.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        self.vector_store.index_to_docstore_id.update(index_to_id)

        if self.persist:
            self.vector_store.save_local(self.index_path)

        return splitted_docs

    def rebuild_index(self, data_path: str = "uploaded_pdfs") -> List[Document]:
        """
        Rebuilds the vector store from scratch using all documents in the given directory.

        Args:
            data_path (str): The path to the directory containing PDF files.

        Returns:
            List[Document]: A list of document chunks that were added to the rebuilt vector store.
        """
        if self.vector_store_path.exists():
            shutil.rmtree(self.vector_store_path)
        self._setup_vector_store()
        documents = self.load_documents(data_path)
        return self.add_documents(documents)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            text = doc.page_content or ""
            if not text.strip():
                continue
            raw_source = doc.metadata.get("source", "unknown")
            source = Path(raw_source).name if raw_source else "unknown"
            page = doc.metadata.get("page", None)
            start = 0
            chunk_index = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if end < len(text):
                    newline_pos = text.rfind("\n", start, end)
                    if newline_pos > start + self.chunk_size // 2:
                        end = newline_pos

                chunk_text = text[start:end].strip()
                if not chunk_text:
                    start += self.chunk_size - self.chunk_overlap
                    continue

                line_start = text.count("\n", 0, start) + 1
                line_end = text.count("\n", 0, end) + 1
                metadata = dict(doc.metadata)
                metadata.update(
                    source=source,
                    page=page,
                    chunk_id=chunk_index + 1,
                    line_start=line_start,
                    line_end=line_end,
                    source_ref=(
                        f"{source}, page {page}, lines {line_start}-{line_end}"
                        if page is not None
                        else f"{source}, lines {line_start}-{line_end}"
                    ),
                )
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
                chunk_index += 1
                start += self.chunk_size - self.chunk_overlap

        return chunks
    
    
    def load_document(self, pdf_path: Path) -> List[Document]:
        """
        Loads and returns the content of a PDF document as a list of Document objects.

        Args:
            pdf_path (Path): The file path to the PDF document.

        Returns:
            List[Document]: A list containing up to the first two Document objects extracted from the PDF.
        """
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        return docs if isinstance(docs, list) else [docs]
    
    def add_document(self, filePath: Path) -> List[Document]:
        """
        Adds a document to the vector store from the specified file path.

        Args:
            filePath (Path): The path to the file to be loaded and added.

        Returns:
            List[Document]: A list of Document objects that were added to the vector store.
        """
        docs = self.load_document(filePath)
        return self.add_documents(docs)
    
    def similarity_search(self, question: str, k:int) -> List[Document]:
        """
        Performs a similarity search on the vector store using the provided question.

        Args:
            question (str): The input query string to search for similar documents.
            k (int): The number of top similar documents to retrieve.

        Returns:
            List[Document]: A list of the top-k documents most similar to the input question.
        """
        return self.vector_store.similarity_search(question, k=k)
    
    def as_retriever(self) -> VectorStoreRetriever:
        """
        Returns a VectorStoreRetriever instance for retrieving documents from the underlying vector store.

        :returns: A VectorStoreRetriever object that can be used to perform retrieval operations.
        :rtype: VectorStoreRetriever
        """
        return self.vector_store.as_retriever()

    def index_websites(self, urls: list[str]) -> List[Document]:
        """
        This method converts the URLs to Document objects, adds them to the vector store,
        and returns the list of Document objects created from the indexed websites.

        Args:
            urls (list[str]): A list of website URLs to be indexed.

        Returns:
            List[Document]: A list of Document objects created from the indexed websites.
        """
        docs = self.website_to_documents(urls)
        return self.add_documents(docs)

    def website_to_documents(self, urls: list[str]) -> list[Document]:
        """
        Loads and parses web pages from the given list of URLs into Document objects.
        Args:
            urls (list[str]): A list of website URLs to load and parse.
        Returns:
            list[Document]: A list of Document objects extracted from the specified web pages.
        Notes:
            Only HTML elements with the classes "post-content", "post-title", or "post-header"
            are parsed from each web page.
        """
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        return docs