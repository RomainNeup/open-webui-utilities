"""
title: Confluence search
description: This tool allows you to search for and retrieve content from Confluence.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
requirements: markdownify, sentence-transformers, numpy, rank_bm25, scikit-learn
version: 0.6.0
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Fix Valves variables
- 0.1.0 - Split Confluence search and Confluence get page
- 0.1.1 - Split Confluence search by title and by content
- 0.1.2 - Improve search by splitting query into words
- 0.1.3 - Add support for Personal Access Token authentication and user settings
- 0.1.4 - Limit setting for search results
- 0.2.0 - Implement RAG (Retrieval Augmented Generation) approach for better search results
- 0.2.1 - Use Open WebUI environment variables
- 0.2.2 - Fix confusion between Confluence API limit and RAG parameters
- 0.2.3 - Memory optimization to prevent OOM errors
- 0.2.4 - Move all hard-coded values to constants
- 0.2.5 - Code structure improvements: search type enum, and better error handling
- 0.2.6 - Add terms splitting option
- 0.3.0 - Add settings for ssl verification
- 0.4.0 - Add support for included/exluded confluence spaces in user settings
- 0.5.0 - Add optional Ollama embedding backend support (local Ollama instance)
- 0.6.0 - Add OpenAI/LiteLLM-compatible embedding backend support
"""

import base64
import json
import requests
import asyncio
import numpy as np
import os
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Any, Optional, Iterable
from pydantic import BaseModel, Field
from markdownify import markdownify
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.neighbors import NearestNeighbors

# New environment variables for Ollama / backend selection
RAG_EMBEDDING_BACKEND = os.environ.get("RAG_EMBEDDING_BACKEND", "sentence_transformers").lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") #change address to match your ollama instance
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "snowflake-arctic-embed2:568m") #change model to the one you use

# Environment variables for OpenAI-compatible embedding API
OPENAI_EMBEDDING_URL = os.environ.get("OPENAI_EMBEDDING_URL", "https://api.openai.com/v1/embeddings")
OPENAI_EMBEDDING_API_KEY = os.environ.get("OPENAI_EMBEDDING_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Get environment variables
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE = (
    os.environ.get("RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE", "True").lower() == "true"
)
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
DEFAULT_RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.0"))
ENABLE_HYBRID_SEARCH = os.environ.get("ENABLE_RAG_HYBRID_SEARCH", "").lower() == "true"
RAG_FULL_CONTEXT = os.environ.get("RAG_FULL_CONTEXT", "False").lower() == "true"

# Memory management settings
MAX_PAGE_SIZE = int(os.environ.get("RAG_FILE_MAX_SIZE", "10000"))
BATCH_SIZE = int(os.environ.get("RAG_FILE_MAX_COUNT", "16"))

# Read cache dir from environment
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/cache")
DEFAULT_MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "sentence_transformers")

# Additional constant values
DEFAULT_RRF_CONSTANT = 60
DEFAULT_DUPLICATE_THRESHOLD = 0.95
DEFAULT_BM25_K = 4
DEFAULT_SIMILARITY_FALLBACK = 0.0

# Field validation constraints
CHUNK_SIZE_MIN = 5
CHUNK_SIZE_MAX = 100000
CHUNK_OVERLAP_MIN = 0
CHUNK_OVERLAP_MAX = 1000
MAX_RESULTS_MIN = 1
SIMILARITY_SCORE_MIN = 0.0
SIMILARITY_SCORE_MAX = 1.0
MAX_PAGE_SIZE_MIN = 1000
MAX_PAGE_SIZE_MAX = 1000000
BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 100

# Default Confluence API related constants
DEFAULT_BASE_URL = "https://example.atlassian.net/wiki"
DEFAULT_USERNAME = "example@example.com"
DEFAULT_API_KEY = "ABCD1234"
DEFAULT_API_RESULT_LIMIT = 5


# Custom exceptions for better error handling
class ConfluenceError(Exception):
    """Base exception for Confluence-related errors"""

    pass


class ConfluenceAuthError(ConfluenceError):
    """Authentication error for Confluence API"""

    pass


class ConfluenceAPIError(ConfluenceError):
    """API error for Confluence endpoints"""

    pass


class ConfluenceModelError(ConfluenceError):
    """Error related to embedding models"""

    pass


# Define an enum for search types
class SearchType(str, Enum):
    """Enum for possible Confluence search types"""

    TITLE = "title"
    CONTENT = "content"
    TITLE_AND_CONTENT = "title_and_content"

    @classmethod
    def from_string(cls, search_type: str) -> "SearchType":
        """Convert string to SearchType enum with error handling"""
        try:
            return cls(search_type.lower())
        except ValueError:
            # Default to title and content if invalid
            return cls.TITLE_AND_CONTENT


@dataclass
class Document:
    """Simple document class to store page content and metadata"""

    page_content: str
    metadata: Dict


class EventEmitter:
    """Helper class to emit events to the UI"""

    def __init__(self, event_emitter: Callable[[dict], Awaitable[None]]):
        self.event_emitter = event_emitter
        pass

    async def emit_status(self, description: str, done: bool, error: bool = False):
        await self.event_emitter(
            {
                "data": {
                    "description": f"{done and (error and '❌' or '✅') or '🔎'} {description}",
                    "status": done and "complete" or "in_progress",
                    "done": done,
                },
                "type": "status",
            }
        )

    async def emit_message(self, content: str):
        await self.event_emitter({"data": {"content": content}, "type": "message"})

    async def emit_source(self, name: str, url: str, content: str, html: bool = False):
        await self.event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": html}],
                    "source": {"name": name},
                },
            }
        )


class TextSplitter:
    """Text splitter for chunking Confluence pages into manageable sections"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap for better context preservation"""
        if not text:
            return []

        # Check for extremely large text and enforce hard limit
        text_length = len(text)
        if text_length > MAX_PAGE_SIZE:
            text_length = MAX_PAGE_SIZE
            text = text[:MAX_PAGE_SIZE]  # Truncate text to hard limit

        chunks = []
        start = 0

        # Process text in smaller windows to reduce memory pressure
        while start < text_length:
            # Calculate window end with extra margin for finding good break points
            window_end = min(start + self.chunk_size + 100, text_length)
            window = text[start:window_end]

            # Find actual chunk end with good break point
            chunk_end = min(self.chunk_size, len(window))
            if chunk_end < len(window):
                # Look for good break points
                search_start = max(self.chunk_overlap, chunk_end - 50)
                for i in range(chunk_end, search_start, -1):
                    if i < len(window) and window[i - 1 : i] in [".", "!", "?", "\n"]:
                        chunk_end = i
                        break

            # Extract chunk from window
            chunk = window[:chunk_end]
            chunks.append(chunk)

            # Update start position for next window
            # Ensure we always make progress by advancing at least 1 character
            # This prevents infinite loops when chunk_end ≤ chunk_overlap
            progress = max(1, chunk_end - self.chunk_overlap)
            start += progress

            # Explicitly clean up the window variable
            window = None

        return chunks

    async def split_documents(
        self, documents: List[Document], event_emitter: EventEmitter
    ) -> List[Document]:
        """Split documents into chunks while preserving metadata"""
        chunked_documents = []
        done = 0
        for doc in documents:
            await event_emitter.emit_status(
                f"Breaking down document {done+1}/{len(documents)} for better analysis",
                False,
            )
            done += 1
            chunks = self.split_text(doc.page_content)
            metadata = dict(doc.metadata)
            for chunk in chunks:
                chunked_documents.append(
                    Document(page_content=chunk, metadata=metadata)
                )
        return chunked_documents


def cosine_similarity(X, Y) -> np.ndarray:
    """Calculate similarity between matrices X and Y"""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = (
        DEFAULT_SIMILARITY_FALLBACK
    )
    return similarity


def filter_similar_embeddings(
    embedded_documents: List[List[float]], similarity_fn: Callable, threshold: float
) -> List[int]:
    """Filter out redundant documents that are too similar to each other"""
    similarity = np.tril(similarity_fn(embedded_documents, embedded_documents), k=-1)
    redundant = np.where(similarity > threshold)
    redundant_stacked = np.column_stack(redundant)
    redundant_sorted = np.argsort(similarity[redundant])[::-1]
    included_idxs = set(range(len(embedded_documents)))
    for first_idx, second_idx in redundant_stacked[redundant_sorted]:
        if first_idx in included_idxs and second_idx in included_idxs:
            # Default to dropping the second document of any highly similar pair.
            included_idxs.remove(second_idx)
    return list(sorted(included_idxs))


class OllamaEmbeddingClient:
    """
    Minimal client for Ollama embedding endpoint.
    Uses POST {host}/api/embeddings with JSON: {"model": "<model>", "prompt": "<text>"}
    It returns a numpy array of shape (n_texts, dim) for list inputs, and 1D array for single string.
    The client parses common response shapes: {"embedding": [...]}, {"embeddings": [...]}, {"data": [{"embedding": [...]}]}
    """

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL, timeout: int = 30):
        self.base_url = host.rstrip("/")
        # endpoint per Ollama docs/examples
        self.endpoint = f"{self.base_url}/api/embeddings"
        self.model = model
        self.session = requests.Session()
        self.timeout = timeout

    def _extract_embedding_from_response(self, resp_json: Any):
        # Try common response shapes
        if isinstance(resp_json, dict):
            # Handles singular (single vector) response
            if "embedding" in resp_json and isinstance(resp_json["embedding"], list):
                return resp_json["embedding"]
            # Handles plural (batch) response
            if "embeddings" in resp_json and isinstance(resp_json["embeddings"], list):
                # could be list of lists or a single list
                embeddings = resp_json["embeddings"]
                if len(embeddings) == 0:
                    return []
                # If embeddings is list of numbers (single embedding), return it
                if all(isinstance(x, (float, int)) for x in embeddings):
                    return embeddings
                # else list of embeddings -> take first if single requested
                if isinstance(embeddings[0], list):
                    return embeddings[0]
            if "data" in resp_json and isinstance(resp_json["data"], list):
                first = resp_json["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]
            # fallback: sometimes a key contains the vector directly
            for v in resp_json.values():
                if (
                    isinstance(v, list)
                    and v
                    and all(isinstance(x, (float, int)) for x in v)
                ):
                    return v
        # If response is top-level list of dicts like [{'embedding': [...]}, ...]
        if isinstance(resp_json, list) and resp_json and isinstance(resp_json[0], dict):
            if "embedding" in resp_json[0]:
                return resp_json[0]["embedding"]
        raise ConfluenceModelError(
            "Unexpected Ollama embeddings response format: "
            + (str(resp_json)[:400] if resp_json is not None else "None")
        )

    def _embed_single(self, text: str) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        try:
            r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ConfluenceModelError(
                f"Failed to call Ollama embeddings API at {self.endpoint}: {e}"
            )
        try:
            resp_json = r.json()
        except ValueError:
            raise ConfluenceModelError(
                f"Ollama embeddings endpoint returned non-JSON response: {r.text[:400]}"
            )
        return self._extract_embedding_from_response(resp_json)

    def encode(self, texts):
        """
        Accepts either a single string or an iterable of strings.
        Returns:
          - for single string: 1D numpy array (dim,)
          - for list of strings: 2D numpy array (n_texts, dim)
        """
        if isinstance(texts, str):
            emb = self._embed_single(texts)
            return np.asarray(emb, dtype=float)
        # assume iterable
        embeddings = []
        for t in texts:
            embeddings.append(self._embed_single(t))
        return np.asarray(embeddings, dtype=float)


class OpenAIEmbeddingClient:
    """
    Client for OpenAI-compatible embedding endpoints (OpenAI, LiteLLM, Azure, etc.).
    Uses POST {url} with JSON: {"model": "<model>", "input": "<text>"}
    """

    def __init__(
        self,
        url: str = OPENAI_EMBEDDING_URL,
        api_key: str = OPENAI_EMBEDDING_API_KEY,
        model: str = OPENAI_EMBEDDING_MODEL,
        timeout: int = 30,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        try:
            r = self.session.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ConfluenceModelError(
                f"Failed to call OpenAI embeddings API at {self.url}: {e}"
            )
        try:
            resp_json = r.json()
        except ValueError:
            raise ConfluenceModelError(
                f"OpenAI embeddings endpoint returned non-JSON response: {r.text[:400]}"
            )
        # Standard OpenAI response: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" in resp_json and isinstance(resp_json["data"], list):
            # Sort by index to ensure correct order
            sorted_data = sorted(resp_json["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]
        raise ConfluenceModelError(
            "Unexpected OpenAI embeddings response format: "
            + (str(resp_json)[:400] if resp_json is not None else "None")
        )

    def encode(self, texts):
        """
        Accepts either a single string or an iterable of strings.
        Returns:
          - for single string: 1D numpy array (dim,)
          - for list of strings: 2D numpy array (n_texts, dim)
        """
        if isinstance(texts, str):
            embeddings = self._embed_batch([texts])
            return np.asarray(embeddings[0], dtype=float)
        embeddings = self._embed_batch(list(texts))
        return np.asarray(embeddings, dtype=float)


class DenseRetriever:
    """Semantic search using document embeddings"""

    def __init__(
        self,
        embedding_model: Any,  # can be SentenceTransformer or OllamaEmbeddingClient or any object with encode(...)
        num_results: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        batch_size: int = BATCH_SIZE,
    ):
        self.embedding_model = embedding_model
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.knn = None
        self.documents = None
        self.document_embeddings = None

    def add_documents(self, documents: List[Document]):
        """Process documents and prepare embeddings for search"""
        self.documents = documents

        # Process documents in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_texts = [doc.page_content for doc in batch]
            # embedding_model.encode should accept list and return 2D array
            batch_embeddings = self.embedding_model.encode(batch_texts)
            # ensure numpy array
            if not isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = np.asarray(batch_embeddings, dtype=float)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        self.document_embeddings = (
            np.vstack(all_embeddings) if all_embeddings else np.array([])
        )

        # Create KNN index
        n_neighbors = min(self.num_results, len(documents)) if len(documents) > 0 else 1
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        if len(self.document_embeddings) > 0:
            self.knn.fit(self.document_embeddings)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Find documents most relevant to the query using semantic similarity"""
        if not self.knn or not self.documents:
            return []

        query_embedding = self.embedding_model.encode(query)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.asarray(query_embedding, dtype=float)

        _, neighbor_indices = self.knn.kneighbors(query_embedding.reshape(1, -1))
        neighbor_indices = neighbor_indices.squeeze(0)

        # Handle case where we have fewer documents than k
        if len(neighbor_indices) == 0:
            return []

        relevant_doc_embeddings = self.document_embeddings[neighbor_indices]

        # Remove duplicative content
        included_idxs = filter_similar_embeddings(
            relevant_doc_embeddings,
            cosine_similarity,
            threshold=DEFAULT_DUPLICATE_THRESHOLD,
        )
        relevant_doc_embeddings = relevant_doc_embeddings[included_idxs]

        # Only include sufficiently relevant documents
        similarity = cosine_similarity([query_embedding], relevant_doc_embeddings)[0]
        similar_enough = np.where(similarity > self.similarity_threshold)[0]
        included_idxs = [included_idxs[i] for i in similar_enough]

        filtered_result_indices = neighbor_indices[included_idxs]

        return [self.documents[i] for i in filtered_result_indices]


def default_preprocessing_func(text: str) -> List[str]:
    """Split text into words for keyword search"""
    return text.split()


class BM25Retriever:
    """Keyword-based retrieval using BM25 algorithm"""

    def __init__(
        self,
        vectorizer: Any,
        docs: List[Document],
        k: int = DEFAULT_BM25_K,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
    ):
        self.vectorizer = vectorizer
        self.docs = docs
        self.k = k
        self.preprocess_func = preprocess_func

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """
        Create a BM25Retriever from a list of Documents.
        """
        texts = [preprocess_func(d.page_content) for d in documents]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts, **bm25_params)
        return cls(
            vectorizer=vectorizer,
            docs=documents,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Find documents most relevant to the query using keyword search"""
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs


def weighted_reciprocal_rank(
    doc_lists: List[List[Document]], weights: List[float], c: int = DEFAULT_RRF_CONSTANT
) -> List[Document]:
    """Combine multiple ranked document lists into a single ranking"""
    if len(doc_lists) != len(weights):
        raise ValueError("Number of rank lists must be equal to the number of weights.")

    # Associate each doc's content with its RRF score
    rrf_score = {}
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            if doc.page_content not in rrf_score:
                rrf_score[doc.page_content] = weight / (rank + c)
            else:
                rrf_score[doc.page_content] += weight / (rank + c)

    # Deduplicate and sort by RRF score
    unique_docs = {}
    all_docs = []
    for doc_list in doc_lists:
        for doc in doc_list:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
                all_docs.append(doc)

    sorted_docs = sorted(
        all_docs,
        reverse=True,
        key=lambda doc: rrf_score[doc.page_content],
    )
    return sorted_docs


class ConfluenceDocumentRetriever:
    """Handles retrieving and processing documents from Confluence"""

    def __init__(
        self,
        model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
        device: str = "cpu",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE,
        embedding_backend: str = RAG_EMBEDDING_BACKEND,
        ollama_host: str = OLLAMA_HOST,
        ollama_model: str = OLLAMA_MODEL,
        openai_url: str = OPENAI_EMBEDDING_URL,
        openai_api_key: str = OPENAI_EMBEDDING_API_KEY,
        openai_model: str = OPENAI_EMBEDDING_MODEL,
    ):
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.text_splitter = TextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        self.embedding_backend = (
            embedding_backend.lower() if embedding_backend else "sentence_transformers"
        )
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.openai_url = openai_url
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

    async def load_embedding_model(self, event_emitter):
        """Load the embedding model for semantic search"""
        await event_emitter.emit_status(
            f"Loading embedding model (backend={self.embedding_backend})...", False
        )

        def load_sentence_transformer():
            return SentenceTransformer(
                self.embedding_model_name,
                cache_folder=self.model_cache_dir,
                device=self.device,
                trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
            )

        # Choose backend
        try:
            if self.embedding_backend in ("ollama", "ollama_api", "ollama_embedding"):
                def load_ollama():
                    return OllamaEmbeddingClient(
                        host=self.ollama_host, model=self.ollama_model
                    )

                self.embedding_model = await asyncio.to_thread(load_ollama)
            elif self.embedding_backend in ("openai", "litellm", "openai_api"):
                def load_openai():
                    return OpenAIEmbeddingClient(
                        url=self.openai_url,
                        api_key=self.openai_api_key,
                        model=self.openai_model,
                    )

                self.embedding_model = await asyncio.to_thread(load_openai)
            else:
                # fallback to sentence transformers
                self.embedding_model = await asyncio.to_thread(
                    load_sentence_transformer
                )

            await event_emitter.emit_status(
                f"Embedding model loaded (backend={self.embedding_backend}).", False
            )
        except Exception as e:
            raise ConfluenceModelError(
                f"Failed to load embedding model ({self.embedding_backend}): {e}"
            )

        return self.embedding_model

    async def retrieve_from_confluence_pages(
        self,
        query: str,
        documents: List[Document],
        event_emitter,
        num_results: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        ensemble_weighting: float = 0.5,
        enable_hybrid_search: bool = ENABLE_HYBRID_SEARCH,
    ) -> List[Document]:
        """Find relevant document chunks from Confluence pages using semantic and keyword search"""
        if not documents:
            return []

        # Chunk documents
        chunked_docs = await self.text_splitter.split_documents(
            documents, event_emitter
        )
        await event_emitter.emit_status(
            f"Prepared {len(chunked_docs)} content sections from {len(documents)} pages for analysis",
            False,
        )

        if not chunked_docs:
            return []

        results = []

        # Determine search approach based on settings
        if not enable_hybrid_search:
            ensemble_weighting = 1.0

        # Semantic search with embeddings
        if ensemble_weighting > 0:
            await event_emitter.emit_status(
                f"Analyzing content meaning in batches of {self.batch_size}...", False
            )
            dense_retriever = DenseRetriever(
                self.embedding_model,
                num_results=num_results,
                similarity_threshold=similarity_threshold,
                batch_size=self.batch_size,
            )
            dense_retriever.add_documents(chunked_docs)
            dense_results = dense_retriever.get_relevant_documents(query)
            await event_emitter.emit_status(
                f"Located {len(dense_results)} sections that match your query's meaning",
                False,
            )
        else:
            dense_results = []

        # Keyword search with BM25
        if ensemble_weighting < 1:
            await event_emitter.emit_status(
                "Looking for matching keywords in content...", False
            )
            keyword_retriever = BM25Retriever.from_documents(
                chunked_docs, k=num_results
            )
            sparse_results = keyword_retriever.get_relevant_documents(query)
            await event_emitter.emit_status(
                f"Found {len(sparse_results)} sections with matching keywords", False
            )
        else:
            sparse_results = []

        # Combine results from both search methods
        results = weighted_reciprocal_rank(
            [dense_results, sparse_results],
            weights=[ensemble_weighting, 1 - ensemble_weighting],
        )

        return results[:num_results]


class Confluence:
    """Interface for Confluence API operations"""

    def __init__(
        self,
        username: str,
        api_key: str,
        base_url: str,
        included_spaces: Optional[List[str]] = None,
        excluded_spaces: Optional[List[str]] = None,
        api_key_auth: bool = True,
        ssl_verify: bool = True
    ):
        self.base_url = base_url
        self.ssl_verify = ssl_verify
        self.headers = self.authenticate(username, api_key, api_key_auth)
        self.included_spaces = included_spaces
        self.excluded_spaces = excluded_spaces

    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a GET request to the Confluence API"""
        url = f"{self.base_url}/rest/api/{endpoint}"
        try:
            response = requests.get(
                url, params=params, headers=self.headers, verify=self.ssl_verify
            )
            if response.status_code == 401:
                raise ConfluenceAuthError(
                    "Authentication failed. Check your credentials."
                )
            elif not response.ok:
                raise ConfluenceAPIError(
                    f"Failed to get data from Confluence: {response.text}"
                )
            return response.json()
        except requests.RequestException as e:
            raise ConfluenceAPIError(
                f"Network error when connecting to Confluence: {str(e)}"
            )

    def _build_search_query(
        self, query: str, field: Optional[str] = None, split_terms: bool = True
    ) -> str:
        """Builds a CQL search query from keywords, optionally limiting to specific fields"""
        terms = query.split()
        if not terms or not split_terms:
            if field:
                return f'{field} ~ "{query}"'
            return f'text ~ "{query}" OR title ~ "{query}"'

        if field:
            cql_terms = " OR ".join([f'{field} ~ "{term}"' for term in terms])
        else:
            cql_terms = " OR ".join(
                [f'title ~ "{term}" OR text ~ "{term}"' for term in terms]
            )
        
        if self.included_spaces is not None and len(self.included_spaces) > 0:
            quoted_spaces = [f"'{space}'" for space in self.included_spaces]
            cql_terms += f' AND space in ({", ".join(quoted_spaces)})'

        if self.excluded_spaces is not None and len(self.excluded_spaces) > 0:
            quoted_excluded_spaces = [f"'{space}'" for space in self.excluded_spaces]
            cql_terms += f' AND space not in ({", ".join(quoted_excluded_spaces)})'
        return cql_terms

    def search_confluence(
        self,
        query: str,
        search_type: SearchType,
        limit: int = 5,
        split_terms: bool = True,
    ) -> List[str]:
        """Unified search method using the search type enum"""
        endpoint = "content/search"

        if search_type == SearchType.TITLE:
            cql_terms = self._build_search_query(query, "title", split_terms)
        elif search_type == SearchType.CONTENT:
            cql_terms = self._build_search_query(query, "text", split_terms)
        else:  # TITLE_AND_CONTENT
            cql_terms = self._build_search_query(query, split_terms=split_terms)

        params = {"cql": f'({cql_terms}) AND type="page"', "limit": limit}
        raw_response = self.get(endpoint, params)
        return [item["id"] for item in raw_response.get("results", [])]

    def get_page(self, page_id: str) -> Dict[str, str]:
        """Get a specific Confluence page by ID"""
        endpoint = f"content/{page_id}"
        params = {"expand": "body.view", "include-version": "false"}
        result = self.get(endpoint, params)

        # Get page content and limit size if needed
        body = markdownify(result["body"]["view"]["value"])
        if len(body) > MAX_PAGE_SIZE:
            body = body[:MAX_PAGE_SIZE]

        return {
            "id": result["id"],
            "title": result["title"],
            "body": body,
            "link": f'{self.base_url}{result["_links"]["webui"]}',
        }

    def authenticate_api_key(self, username: str, api_key: str) -> Dict[str, str]:
        """Create auth headers using username and API key"""
        auth_string = f"{username}:{api_key}"
        encoded_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )
        return {"Authorization": "Basic " + encoded_auth_string}

    def authenticate_personal_access_token(self, access_token: str) -> Dict[str, str]:
        """Create auth headers using personal access token"""
        return {"Authorization": f"Bearer {access_token}"}

    def authenticate(
        self, username: str, api_key: str, api_key_auth: bool
    ) -> Dict[str, str]:
        """Set up authentication based on configuration"""
        if api_key_auth:
            return self.authenticate_api_key(username, api_key)
        else:
            return self.authenticate_personal_access_token(api_key)


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        self.document_retriever = None

    class Valves(BaseModel):
        """Configuration options for the Confluence search tool"""

        base_url: str = Field(
            DEFAULT_BASE_URL,
            description="The base URL of your Confluence instance",
        )
        ssl_verify: bool = Field(
            True,
            description="SSL verification"
        )
        username: str = Field(
            DEFAULT_USERNAME,
            description="Default username (leave empty for personal access token)",
        )
        api_key: str = Field(
            DEFAULT_API_KEY, description="Default API key or personal access token"
        )
        api_result_limit: int = Field(
            DEFAULT_API_RESULT_LIMIT,
            description="Maximum number of pages to retrieve from Confluence API",
            required=True,
        )
        embedding_model_save_path: str = Field(
            DEFAULT_MODEL_CACHE_DIR,
            description="Path to the folder in which embedding models will be saved",
        )
        embedding_backend: str = Field(
            default=RAG_EMBEDDING_BACKEND,
            description='Which embedding backend to use. Allowed values: "sentence_transformers", "ollama", "openai".',
        )
        embedding_model_name: str = Field(
            DEFAULT_EMBEDDING_MODEL,
            description="Name or path of the embedding model to use (used for sentence-transformers backend).",
        )
        ollama_host: str = Field(
            default=OLLAMA_HOST,
            description="Base URL of local Ollama server (used when embedding_backend='ollama')",
        )
        ollama_model_name: str = Field(
            default=OLLAMA_MODEL,
            description="Name of the Ollama embedding model (used when embedding_backend='ollama')",
        )
        openai_embedding_url: str = Field(
            default=OPENAI_EMBEDDING_URL,
            description="URL of the OpenAI-compatible embeddings endpoint (used when embedding_backend='openai')",
        )
        openai_embedding_api_key: str = Field(
            default=OPENAI_EMBEDDING_API_KEY,
            description="API key for the OpenAI-compatible embeddings endpoint (used when embedding_backend='openai')",
        )
        openai_embedding_model: str = Field(
            default=OPENAI_EMBEDDING_MODEL,
            description="Model name for the OpenAI-compatible embeddings endpoint (used when embedding_backend='openai')",
        )
        cpu_only: bool = Field(
            default=True, 
            description="Run the tool on CPU only"
        )
        chunk_size: int = Field(
            default=DEFAULT_CHUNK_SIZE,
            description="Max. chunk size for Confluence pages",
            ge=CHUNK_SIZE_MIN,
            le=CHUNK_SIZE_MAX,
        )
        chunk_overlap: int = Field(
            default=DEFAULT_CHUNK_OVERLAP,
            description="Overlap size between chunks",
            ge=CHUNK_OVERLAP_MIN,
            le=CHUNK_OVERLAP_MAX,
        )
        max_results: int = Field(
            default=DEFAULT_TOP_K,
            description="Maximum number of relevant chunks to return after RAG processing",
            ge=MAX_RESULTS_MIN,
        )
        similarity_threshold: float = Field(
            default=DEFAULT_RELEVANCE_THRESHOLD,
            description="Similarity Score Threshold. "
            "Discard chunks that are not similar enough to the "
            "search query and hence fall below the threshold.",
            ge=SIMILARITY_SCORE_MIN,
            le=SIMILARITY_SCORE_MAX,
        )
        ensemble_weighting: float = Field(
            default=0.5,
            description="Ensemble Weighting. "
            "Smaller values = More keyword oriented, Larger values = More focus on semantic similarity. "
            "Ignored if hybrid search is disabled.",
            ge=SIMILARITY_SCORE_MIN,
            le=SIMILARITY_SCORE_MAX,
        )
        enable_hybrid_search: bool = Field(
            default=ENABLE_HYBRID_SEARCH,
            description="Enable hybrid search (combine semantic and keyword search)",
        )
        full_context: bool = Field(
            default=RAG_FULL_CONTEXT,
            description="Return full document content instead of just the most relevant chunks",
        )
        max_page_size: int = Field(
            default=MAX_PAGE_SIZE,
            description="Maximum size in characters for a Confluence page to prevent OOM",
            ge=MAX_PAGE_SIZE_MIN,
            le=MAX_PAGE_SIZE_MAX,
        )
        batch_size: int = Field(
            default=BATCH_SIZE,
            description="Number of documents to process at once for embedding",
            ge=BATCH_SIZE_MIN,
            le=BATCH_SIZE_MAX,
        )
        pass

    class UserValves(BaseModel):
        """User-specific configuration options"""

        api_key_auth: bool = Field(
            True,
            description="Use API key authentication; disable this to use a personal access token instead.",
        )
        username: str = Field(
            "",
            description="Username, typically your email address; leave empty if using a personal access token or default settings.",
        )
        api_key: str = Field(
            "",
            description="API key or personal access token; leave empty to use the default settings.",
        )
        split_terms: bool = Field(
            True,
            description="Split search query into words for better search results.",
        )
        included_confluence_spaces: str = Field(
            "",
            description="Comma-separated list of Confluence spaces to search in; leave empty to search all spaces.",
        )
        excluded_confluence_spaces: str = Field(
            "",
            description="Comma-separated list of Confluence spaces to exclude from the search; leave empty to include all spaces.",
        )
        pass

    # Get content from Confluence
    async def search_confluence(
        self,
        query: str,
        type: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ) -> str:
        """
        Search for a query on Confluence. This returns the result of the search on Confluence.
        Use it to search for a query on Confluence. When a user mentions a search on Confluence, this must be used.
        It can search by content or by title.
        Note: This returns a list of pages that match the search query.
        :param query: The text to search for on Confluence or the title of the page if asked to search by title. MUST be a string.
        :param type: The type of search to perform ('content' or 'title' or 'title_and_content')
        :return: A list of search results from Confluence in JSON format (id, title, body, link). If no results are found, an empty list is returned.
        """
        event_emitter = EventEmitter(__event_emitter__)

        try:
            # Convert the search type string to enum for better validation
            search_type = SearchType.from_string(type)

            # Get the username and API key
            if __user__ and "valves" in __user__:
                user_valves = __user__["valves"]
                api_key_auth = user_valves.api_key_auth
                api_username = user_valves.username or self.valves.username
                api_key = user_valves.api_key or self.valves.api_key
                split_terms = user_valves.split_terms
                included_confluence_spaces = (
                    user_valves.included_confluence_spaces.split(",")
                    if user_valves.included_confluence_spaces
                    else None
                )
                if included_confluence_spaces:
                    included_confluence_spaces = [
                        space.strip() for space in included_confluence_spaces
                    ]
                excluded_confluence_spaces = (
                    user_valves.excluded_confluence_spaces.split(",")
                    if user_valves.excluded_confluence_spaces
                    else None
                )
                if excluded_confluence_spaces:
                    excluded_confluence_spaces = [
                        space.strip() for space in excluded_confluence_spaces
                    ]
            else:
                api_username = self.valves.username
                api_key = self.valves.api_key
                api_key_auth = True
                split_terms = True

            if (api_key_auth and not api_username) or not api_key:
                await event_emitter.emit_status(
                    "Please provide a username and API key or personal access token.",
                    True,
                    True,
                )
                return "Error: Please provide a username and API key or personal access token."

            # Apply memory settings from valves
            global MAX_PAGE_SIZE, BATCH_SIZE
            MAX_PAGE_SIZE = self.valves.max_page_size
            BATCH_SIZE = self.valves.batch_size

            # Ensure cache directory exists
            model_cache_dir = (
                self.valves.embedding_model_save_path or DEFAULT_MODEL_CACHE_DIR
            )
            try:
                os.makedirs(model_cache_dir, exist_ok=True)
            except Exception as e:
                await event_emitter.emit_status(
                    f"Error creating model cache directory: {str(e)}", True, True
                )
                return f"Error: {str(e)}"

            # Initialize document retriever and load model with proper error handling
            try:
                if not self.document_retriever:
                    self.document_retriever = ConfluenceDocumentRetriever(
                        model_cache_dir=model_cache_dir,
                        device="cpu" if self.valves.cpu_only else "cuda",
                        embedding_model_name=self.valves.embedding_model_name,
                        batch_size=BATCH_SIZE,
                        embedding_backend=self.valves.embedding_backend,
                        ollama_host=self.valves.ollama_host,
                        ollama_model=self.valves.ollama_model_name,
                        openai_url=self.valves.openai_embedding_url,
                        openai_api_key=self.valves.openai_embedding_api_key,
                        openai_model=self.valves.openai_embedding_model,
                    )

                if not self.document_retriever.embedding_model:
                    try:
                        await self.document_retriever.load_embedding_model(event_emitter)
                    except ConfluenceModelError as e:
                        await event_emitter.emit_status(
                            f"Error loading embedding model: {str(e)}", True, True
                        )
                        return f"Error: Failed to load embedding model: {str(e)}"
            except Exception as e:
                await event_emitter.emit_status(
                    f"Error initializing document retriever: {str(e)}", True, True
                )
                return f"Error: Failed to initialize document retriever: {str(e)}"

            # Create Confluence client with error handling
            try:
                confluence = Confluence(
                    username=api_username,
                    api_key=api_key,
                    base_url=self.valves.base_url,
                    api_key_auth=api_key_auth,
                    ssl_verify=self.valves.ssl_verify,
                    included_spaces=included_confluence_spaces,
                    excluded_spaces=excluded_confluence_spaces,
                )
            except ConfluenceAuthError as e:
                await event_emitter.emit_status(
                    f"Authentication error: {str(e)}", True, True
                )
                return f"Error: Authentication failed: {str(e)}"

            await event_emitter.emit_status(
                f"Searching Confluence for '{query}' in {search_type.value}...", False
            )

            # Search using the Confluence API
            try:
                searchResponse = confluence.search_confluence(
                    query, search_type, self.valves.api_result_limit, split_terms
                )
            except ConfluenceAPIError as e:
                await event_emitter.emit_status(
                    f"API error during search: {str(e)}", True, True
                )
                return f"Error: Confluence API error: {str(e)}"

            if not searchResponse:
                await event_emitter.emit_status(
                    f"No matching results found in Confluence for '{query}'", True
                )
                return json.dumps([])

            # Fetch the full content of each page found
            raw_documents = []
            for i, item in enumerate(searchResponse):
                await event_emitter.emit_status(
                    f"Retrieving Confluence page {i+1}/{len(searchResponse)}...", False
                )
                page = confluence.get_page(item)
                raw_documents.append(
                    Document(
                        page_content=page["body"],
                        metadata={
                            "id": page["id"],
                            "title": page["title"],
                            "source": page["link"],
                        },
                    )
                )

            # If full context mode is enabled, skip RAG processing and return complete pages
            if self.valves.full_context:
                await event_emitter.emit_status(
                    f"Preparing all {len(raw_documents)} Confluence pages for you...",
                    False,
                )

                # Create results with the complete content
                results = []
                for doc in raw_documents:
                    result = {
                        "id": doc.metadata["id"],
                        "title": doc.metadata["title"],
                        "body": doc.page_content,
                        "link": doc.metadata["source"],
                    }

                    # Add citations for each full document
                    await event_emitter.emit_source(
                        result["title"], result["link"], result["body"]
                    )

                    results.append(result)

                await event_emitter.emit_status(
                    f"Found {len(results)} complete Confluence pages matching '{query}'",
                    True,
                )

                return json.dumps(results)

            # Apply RAG processing to find the most relevant content
            elif raw_documents:
                await event_emitter.emit_status(
                    f"Analyzing {len(raw_documents)} pages to find the most relevant content...",
                    False,
                )

                # Update text splitter settings before processing
                self.document_retriever.text_splitter.chunk_size = (
                    self.valves.chunk_size
                )
                self.document_retriever.text_splitter.chunk_overlap = (
                    self.valves.chunk_overlap
                )

                relevant_chunks = (
                    await self.document_retriever.retrieve_from_confluence_pages(
                        query=query,
                        documents=raw_documents,
                        event_emitter=event_emitter,
                        num_results=self.valves.max_results,
                        similarity_threshold=self.valves.similarity_threshold,
                        ensemble_weighting=self.valves.ensemble_weighting,
                        enable_hybrid_search=self.valves.enable_hybrid_search,
                    )
                )

                # Clear raw documents to free memory
                raw_documents = None

                # Group chunks by page to build results
                page_chunks = {}
                for chunk in relevant_chunks:
                    page_id = chunk.metadata["id"]
                    if page_id not in page_chunks:
                        page_chunks[page_id] = {
                            "id": page_id,
                            "title": chunk.metadata["title"],
                            "link": chunk.metadata["source"],
                            "chunks": [],
                        }
                    page_chunks[page_id]["chunks"].append(chunk.page_content)

                # Create final results
                results = []
                for page_id, page_data in page_chunks.items():
                    # Join chunks with context into a coherent body
                    body = "\n\n".join(page_data["chunks"])

                    result = {
                        "id": page_data["id"],
                        "title": page_data["title"],
                        "body": body,
                        "link": page_data["link"],
                    }

                    # Add citations for each relevant chunk
                    await event_emitter.emit_source(
                        result["title"], result["link"], result["body"]
                    )

                    results.append(result)
            else:
                results = []

            await event_emitter.emit_status(
                f"Search complete! Found {len(results)} pages with information about '{query}'",
                True,
            )

            return json.dumps(results)
        except Exception as e:
            await event_emitter.emit_status(
                f"Unexpected error during search: {str(e)}.", True, True
            )
            return f"Error: {str(e)}"
