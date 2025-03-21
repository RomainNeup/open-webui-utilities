"""
title: Confluence search
description: This tool allows you to search for and retrieve content from Confluence.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
requirements: markdownify, sentence-transformers, numpy, rank_bm25, scikit-learn
version: 0.2.2
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Fix Valves variables
- 0.1.0 - Split Confuence search and Confluence get page
- 0.1.1 - Split Confluence search by title and by content
- 0.1.2 - Improve search by splitting query into words
- 0.1.3 - Add support for Personal Access Token authentication and user settings
- 0.1.4 - Limit setting for search results
- 0.2.0 - Implement RAG (Retrieval Augmented Generation) approach for better search results
- 0.2.1 - Use Open WebUI environment variables
- 0.2.2 - Fix confusion between Confluence API limit and RAG parameters
"""

import base64
import json
import requests
import asyncio
import copy
import numpy as np
import os
from typing import Awaitable, Callable, Dict, List, Any, Optional, Iterable
from pydantic import BaseModel, Field
from markdownify import markdownify
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.neighbors import NearestNeighbors

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
DEFAULT_API_RESULT_LIMIT = 5  # Default number of results to return from Confluence API
DEFAULT_RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.0"))
ENABLE_HYBRID_SEARCH = os.environ.get("ENABLE_RAG_HYBRID_SEARCH", "").lower() == "true"
RAG_FULL_CONTEXT = os.environ.get("RAG_FULL_CONTEXT", "False").lower() == "true"

# Read cache dir from environment
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/cache")
DEFAULT_MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "sentence_transformers")


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

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            # Find a good breaking point at sentence boundaries
            if end < text_len:
                for i in range(end, max(start + self.chunk_overlap, end - 50), -1):
                    if i < text_len and text[i] in [".", "!", "?", "\n"]:
                        end = i + 1
                        break

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata"""
        chunked_documents = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                metadata = copy.deepcopy(doc.metadata)
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
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
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


class DenseRetriever:
    """Semantic search using document embeddings"""

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        num_results: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.embedding_model = embedding_model
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.knn = NearestNeighbors(n_neighbors=num_results)
        self.documents = None
        self.document_embeddings = None

    def add_documents(self, documents: List[Document]):
        """Process documents and prepare embeddings for search"""
        self.documents = documents
        self.document_embeddings = self.embedding_model.encode(
            [doc.page_content for doc in documents]
        )
        self.knn.fit(self.document_embeddings)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Find documents most relevant to the query using semantic similarity"""
        query_embedding = self.embedding_model.encode(query)

        _, neighbor_indices = self.knn.kneighbors(query_embedding.reshape(1, -1))
        neighbor_indices = neighbor_indices.squeeze(0)

        relevant_doc_embeddings = self.document_embeddings[neighbor_indices]
        # Remove duplicative content
        included_idxs = filter_similar_embeddings(
            relevant_doc_embeddings, cosine_similarity, threshold=0.95
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
        k: int = 4,
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
    doc_lists: List[List[Document]], weights: List[float], c: int = 60
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
    ):
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name
        self.text_splitter = TextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )

    async def load_embedding_model(self, event_emitter):
        """Load the embedding model for semantic search"""
        await event_emitter.emit_status(
            f"Loading embedding model {self.embedding_model_name}...", False
        )

        def load_model():
            return SentenceTransformer(
                self.embedding_model_name,
                cache_folder=self.model_cache_dir,
                device=self.device,
                trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
            )

        # Run in an executor to avoid blocking the event loop
        self.embedding_model = await asyncio.to_thread(load_model)

        return self.embedding_model

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documents into smaller chunks for better processing"""
        return self.text_splitter.split_documents(documents)

    def retrieve_from_confluence_pages(
        self,
        query: str,
        documents: List[Document],
        num_results: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        ensemble_weighting: float = 0.5,
        enable_hybrid_search: bool = ENABLE_HYBRID_SEARCH,
    ) -> List[Document]:
        """Find relevant document chunks from Confluence pages using semantic and keyword search"""
        if not documents:
            return []

        chunked_docs = self.chunk_documents(documents)

        if not chunked_docs:
            return []

        results = []

        # Determine search approach based on settings
        if not enable_hybrid_search:
            ensemble_weighting = 1.0

        # Semantic search with embeddings
        if ensemble_weighting > 0:
            dense_retriever = DenseRetriever(
                self.embedding_model,
                num_results=num_results,
                similarity_threshold=similarity_threshold,
            )
            dense_retriever.add_documents(chunked_docs)
            dense_results = dense_retriever.get_relevant_documents(query)
        else:
            dense_results = []

        # Keyword search with BM25
        if ensemble_weighting < 1:
            keyword_retriever = BM25Retriever.from_documents(
                chunked_docs, k=num_results
            )
            sparse_results = keyword_retriever.get_relevant_documents(query)
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
        self, username: str, api_key: str, base_url: str, api_key_auth: bool = True
    ):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key, api_key_auth)
        pass

    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a GET request to the Confluence API"""
        url = f"{self.base_url}/rest/api/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise Exception(f"Failed to get data from Confluence: {response.text}")
        return response.json()

    def search_by_title(self, query: str, limit: int = 5) -> List[str]:
        """Search Confluence pages by title, returning page IDs"""
        endpoint = "content/search"
        # Make each word optional to improve search results
        terms = query.split()
        if terms:
            cql_terms = " OR ".join([f'title ~ "{term}"' for term in terms])
        else:
            cql_terms = f'title ~ "{query}"'
        params = {"cql": f'({cql_terms}) AND type="page"', "limit": limit}
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["results"]:
            response.append(item["id"])
        return response

    def search_by_content(self, query: str, limit: int = 5) -> List[str]:
        """Search Confluence pages by content, returning page IDs"""
        endpoint = "content/search"
        # Split query into individual terms and join them with OR such that each word is optional
        terms = query.split()
        if terms:
            cql_terms = " OR ".join([f'text ~ "{term}"' for term in terms])
        else:
            cql_terms = f'text ~ "{query}"'
        params = {"cql": f'({cql_terms}) AND type="page"', "limit": limit}
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["results"]:
            response.append(item["id"])
        return response

    def search_by_title_and_content(self, query: str, limit: int = 5) -> List[str]:
        """Search Confluence pages by both title and content, returning page IDs"""
        endpoint = "content/search"
        # Split query into words and join them with OR; each word is optional.
        terms = query.split()
        if terms:
            cql_terms = " OR ".join(
                [f'title ~ "{term}" OR text ~ "{term}"' for term in terms]
            )
        else:
            cql_terms = f'title ~ "{query}" OR text ~ "{query}"'
        params = {"cql": f'({cql_terms}) AND type="page"', "limit": limit}
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["results"]:
            response.append(item["id"])
        return response

    def get_page(self, page_id: str) -> Dict[str, str]:
        """Get a specific Confluence page by ID"""
        endpoint = f"content/{page_id}"
        params = {"expand": "body.view", "include-version": "false"}
        result = self.get(endpoint, params)
        return {
            "id": result["id"],
            "title": result["title"],
            "body": markdownify(result["body"]["view"]["value"]),
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
        pass

    class Valves(BaseModel):
        """Configuration options for the Confluence search tool"""

        base_url: str = Field(
            "https://example.atlassian.net/wiki",
            description="The base URL of your Confluence instance",
        )
        username: str = Field(
            "example@example.com",
            description="Default username (leave empty for personal access token)",
        )
        api_key: str = Field(
            "ABCD1234", description="Default API key or personal access token"
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
        embedding_model_name: str = Field(
            DEFAULT_EMBEDDING_MODEL,
            description="Name or path of the embedding model to use",
        )
        cpu_only: bool = Field(default=True, description="Run the tool on CPU only")
        chunk_size: int = Field(
            default=DEFAULT_CHUNK_SIZE,
            description="Max. chunk size for Confluence pages",
            ge=5,
            le=100000,
        )
        chunk_overlap: int = Field(
            default=DEFAULT_CHUNK_OVERLAP,
            description="Overlap size between chunks",
            ge=0,
            le=1000,
        )
        max_results: int = Field(
            default=DEFAULT_TOP_K,
            description="Maximum number of relevant chunks to return after RAG processing",
            ge=1,
        )
        similarity_threshold: float = Field(
            default=DEFAULT_RELEVANCE_THRESHOLD,
            description="Similarity Score Threshold. "
            "Discard chunks that are not similar enough to the "
            "search query and hence fall below the threshold.",
            ge=0.0,
            le=1.0,
        )
        ensemble_weighting: float = Field(
            default=0.5,
            description="Ensemble Weighting. "
            "Smaller values = More keyword oriented, Larger values = More focus on semantic similarity. "
            "Ignored if hybrid search is disabled.",
            ge=0.0,
            le=1.0,
        )
        enable_hybrid_search: bool = Field(
            default=ENABLE_HYBRID_SEARCH,
            description="Enable hybrid search (combine semantic and keyword search)",
        )
        full_context: bool = Field(
            default=RAG_FULL_CONTEXT,
            description="Return full document content instead of just the most relevant chunks",
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

        # Get the username and API key
        if __user__ and "valves" in __user__:
            user_valves = __user__["valves"]
            api_key_auth = user_valves.api_key_auth
            api_username = user_valves.username or self.valves.username
            api_key = user_valves.api_key or self.valves.api_key
        else:
            api_username = self.valves.username
            api_key = self.valves.api_key
            api_key_auth = True

        if (api_key_auth and not api_username) or not api_key:
            await event_emitter.emit_status(
                "Please provide a username and API key or personal access token.",
                True,
                True,
            )
            return (
                "Error: Please provide a username and API key or personal access token."
            )

        # Check if embedding model path is set
        model_cache_dir = self.valves.embedding_model_save_path
        if not model_cache_dir:
            model_cache_dir = DEFAULT_MODEL_CACHE_DIR
            # Try to create the directory if it doesn't exist
            try:
                os.makedirs(model_cache_dir, exist_ok=True)
            except Exception as e:
                await event_emitter.emit_status(
                    f"Error creating model cache directory: {str(e)}", True, True
                )
                return f"Error: {str(e)}"

        # Initialize document retriever if not already done
        if not self.document_retriever:
            self.document_retriever = ConfluenceDocumentRetriever(
                model_cache_dir=model_cache_dir,
                device="cpu" if self.valves.cpu_only else "cuda",
                embedding_model_name=self.valves.embedding_model_name,
            )

        # Load the embedding model if not already loaded
        if not self.document_retriever.embedding_model:
            await self.document_retriever.load_embedding_model(event_emitter)

        confluence = Confluence(
            api_username, api_key, self.valves.base_url, api_key_auth
        )

        search_type = type.lower()

        await event_emitter.emit_status(
            f"Searching for {search_type} '{query}' on Confluence...", False
        )
        try:
            # Search using the Confluence API
            if search_type == "title":
                searchResponse = confluence.search_by_title(
                    query, self.valves.api_result_limit
                )
            elif search_type == "content":
                searchResponse = confluence.search_by_content(
                    query, self.valves.api_result_limit
                )
            else:
                searchResponse = confluence.search_by_title_and_content(
                    query, self.valves.api_result_limit
                )

            # Fetch the full content of each page found
            raw_documents = []
            for item in searchResponse:
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
                    f"Returning {len(raw_documents)} full Confluence pages (RAG disabled)...",
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
                    f"Search for {search_type} '{query}' on Confluence complete. ({len(results)} full pages found)",
                    True,
                )

                return json.dumps(results)

            # Apply RAG processing to find the most relevant content
            elif raw_documents:
                await event_emitter.emit_status(
                    f"Processing {len(raw_documents)} Confluence pages with chunk size {self.valves.chunk_size}...",
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
                    self.document_retriever.retrieve_from_confluence_pages(
                        query=query,
                        documents=raw_documents,
                        num_results=self.valves.max_results,
                        similarity_threshold=self.valves.similarity_threshold,
                        ensemble_weighting=self.valves.ensemble_weighting,
                        enable_hybrid_search=self.valves.enable_hybrid_search,
                    )
                )

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
                f"Search for {search_type} '{query}' on Confluence complete. ({len(results)} pages with relevant content found)",
                True,
            )

            return json.dumps(results)
        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to search for {search_type} '{query}': {e}.", True, True
            )
            return f"Error: {str(e)}"
