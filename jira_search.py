"""
title: Jira Search
description: This tool allows you to search issues from Jira.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
requirements: markdownify, sentence-transformers, numpy, rank_bm25, scikit-learn
version: 0.3.0
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Implement Jira search
- 0.1.0 - Add support for Personal Access Token authentication and user settings
- 0.1.1 - Limit setting for search results
- 0.1.2 - Add terms splitting option
- 0.2.0 - Add setting for SSL verification
- 0.3.0 - Implement RAG (Retrieval Augmented Generation) for better search results
"""

import base64
import json
import os
import asyncio
import numpy as np
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Iterable
from dataclasses import dataclass
import requests
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.neighbors import NearestNeighbors
import logging

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
MAX_ISSUE_SIZE = int(os.environ.get("RAG_FILE_MAX_SIZE", "10000"))
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
MAX_ISSUE_SIZE_MIN = 1000
MAX_ISSUE_SIZE_MAX = 1000000
BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 100


class EventEmitter:
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


@dataclass
class Document:
    """Simple document class to store issue content and metadata"""

    issue_content: str
    metadata: Dict


class TextSplitter:
    """Text splitter for chunking Jira issues into manageable sections"""

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
        if text_length > MAX_ISSUE_SIZE:
            text_length = MAX_ISSUE_SIZE
            text = text[:MAX_ISSUE_SIZE]  # Truncate text to hard limit

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
                f"Breaking down issue {done+1}/{len(documents)} for better analysis",
                False,
            )
            done += 1
            chunks = self.split_text(doc.issue_content)
            metadata = dict(doc.metadata)
            for chunk in chunks:
                chunked_documents.append(
                    Document(issue_content=chunk, metadata=metadata)
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


class DenseRetriever:
    """Semantic search using document embeddings"""

    def __init__(
        self,
        embedding_model: SentenceTransformer,
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
            batch_texts = [doc.issue_content for doc in batch]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        self.document_embeddings = (
            np.vstack(all_embeddings) if all_embeddings else np.array([])
        )

        # Create KNN index
        self.knn = NearestNeighbors(n_neighbors=min(self.num_results, len(documents)))
        if len(self.document_embeddings) > 0:
            self.knn.fit(self.document_embeddings)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Find documents most relevant to the query using semantic similarity"""
        if not self.knn or not self.documents:
            return []

        query_embedding = self.embedding_model.encode(query)

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
        texts = [preprocess_func(d.issue_content) for d in documents]
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
            if doc.issue_content not in rrf_score:
                rrf_score[doc.issue_content] = weight / (rank + c)
            else:
                rrf_score[doc.issue_content] += weight / (rank + c)

    # Deduplicate and sort by RRF score
    unique_docs = {}
    all_docs = []
    for doc_list in doc_lists:
        for doc in doc_list:
            if doc.issue_content not in unique_docs:
                unique_docs[doc.issue_content] = doc
                all_docs.append(doc)

    sorted_docs = sorted(
        all_docs,
        reverse=True,
        key=lambda doc: rrf_score[doc.issue_content],
    )
    return sorted_docs


class JiraDocumentRetriever:
    """Handles retrieving and processing documents from Jira"""

    def __init__(
        self,
        model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
        device: str = "cpu",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE,
    ):
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
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

    async def retrieve_from_jira_issues(
        self,
        query: str,
        documents: List[Document],
        event_emitter,
        num_results: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        ensemble_weighting: float = 0.5,
        enable_hybrid_search: bool = ENABLE_HYBRID_SEARCH,
    ) -> List[Document]:
        """Find relevant document chunks from Jira issues using semantic and keyword search"""
        if not documents:
            return []

        # Chunk documents
        chunked_docs = await self.text_splitter.split_documents(
            documents, event_emitter
        )
        await event_emitter.emit_status(
            f"Prepared {len(chunked_docs)} content sections from {len(documents)} issues for analysis",
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


class Jira:
    def __init__(
        self,
        username: str,
        api_key: str,
        base_url: str,
        api_key_auth: bool = True,
        ssl_verify: bool = True,
    ):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key, api_key_auth)
        self.ssl_verify = ssl_verify
        self.logger = logging.getLogger("jira_search")
        pass

    def get(self, endpoint: str, params: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        self.logger.debug(f"Jira API Request: {url}")
        self.logger.debug(f"Headers: {self.headers}")
        self.logger.debug(f"Params: {params}")

        try:
            response = requests.get(
                url, params=params, headers=self.headers, verify=self.ssl_verify
            )

            self.logger.debug(f"Jira API Response Status: {response.status_code}")
            self.logger.debug(f"Response Headers: {response.headers}")
            self.logger.debug(f"Response Body: {response.text}")

            return response.json()
        except Exception as e:
            self.logger.error(f"Jira API Request failed: {e}")
            raise

    def search(
        self, query: str, limit: int = 5, split_terms: bool = True
    ) -> Dict[str, Any]:
        # Use the new JQL endpoint instead of the deprecated one
        endpoint = "search/jql"
        params = {
            "jql": query,
            "maxResults": limit,
            "fields": [
                "summary",
                "description",
                "status",
                "issuetype",
                "priority",
                "assignee",
                "created",
            ],
        }
        raw_response = self.get(endpoint, params)
        
        # Extract issues from the response
        if "issues" not in raw_response:
            return {"issues": []}
            
        return raw_response

    def get_issue(self, issue_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific issue"""
        endpoint = f"issue/{issue_key}"
        params = {
            "fields": "summary,description,created,status,issuetype,priority,assignee"
        }
        result = self.get(endpoint, params)

        # Extract relevant information
        fields = result.get("fields", {})
        description = fields.get("description", {})

        # Convert description to plain text if it exists
        content = ""
        if description and "content" in description:
            for item in description.get("content", []):
                if "content" in item:
                    for text_item in item.get("content", []):
                        if text_item.get("type") == "text":
                            content += text_item.get("text", "") + " "

        return {
            "key": result.get("key", ""),
            "summary": fields.get("summary", ""),
            "description": content,
            "status": (
                fields.get("status", {}).get("name", "") if fields.get("status") else ""
            ),
            "issuetype": (
                fields.get("issuetype", {}).get("name", "")
                if fields.get("issuetype")
                else ""
            ),
            "priority": (
                fields.get("priority", {}).get("name", "")
                if fields.get("priority")
                else ""
            ),
            "assignee": (
                fields.get("assignee", {}).get("displayName", "")
                if fields.get("assignee")
                else ""
            ),
            "created": fields.get("created", ""),
            "link": f"{self.base_url}/browse/{result.get('key', '')}",
        }

    def authenticate_api_key(self, username: str, api_key: str) -> Dict[str, str]:
        auth_string = f"{username}:{api_key}"
        encoded_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )
        return {"Authorization": "Basic " + encoded_auth_string}

    def authenticate_personal_access_token(self, access_token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {access_token}"}

    def authenticate(
        self, username: str, api_key: str, api_key_auth: bool
    ) -> Dict[str, str]:
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
        base_url: str = Field(
            "https://example.atlassian.net",
            description="The base URL of your Jira instance",
        )
        username: str = Field(
            "example@example.com",
            description="Default username (leave empty for personal access token)",
        )
        api_key: str = Field(
            "ABCD1234", description="Default API key or personal access token"
        )
        ssl_verify: bool = Field(True, description="SSL verification")
        result_limit: int = Field(
            5,
            description="The maximum number of search results to return",
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
            description="Max. chunk size for Jira issues",
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
            description="Return full issue content instead of just the most relevant chunks",
        )
        max_issue_size: int = Field(
            default=MAX_ISSUE_SIZE,
            description="Maximum size in characters for a Jira issue to prevent OOM",
            ge=MAX_ISSUE_SIZE_MIN,
            le=MAX_ISSUE_SIZE_MAX,
        )
        batch_size: int = Field(
            default=BATCH_SIZE,
            description="Number of documents to process at once for embedding",
            ge=BATCH_SIZE_MIN,
            le=BATCH_SIZE_MAX,
        )
        debug: bool = Field(
            False,
            description="Enable debug logging for Jira API requests and responses",
        )
        pass

    class UserValves(BaseModel):
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
        split_query: bool = Field(
            True,
            description="Split the query into individual words and search for each word separately.",
        )
        pass

    async def search_jira(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ) -> str:
        """
        Search for a query on Jira. This returns the result of the search on Jira.
        Use it to search for a query on Jira. When a user mentions a search on Jira, this must be used.
        Note: This returns a list of issues that match the search query.
        :param query: The text to search for on Jira. MUST be a string.
        :return: A list of search results from Jira in JSON format with detailed information. If no results are found, an empty list is returned.
        """
        event_emitter = EventEmitter(__event_emitter__)

        # Get the logger instance (OpenWebUI already configures this)
        logger = logging.getLogger("jira_search")

        # Set debug level if valves.debug is True
        if self.valves.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Ensure the logger doesn't propagate to root (optional, but safe)
        logger.propagate = False

        try:
            # Get the username and API key
            if __user__ and "valves" in __user__:
                user_valves = __user__["valves"]
                api_key_auth = user_valves.api_key_auth
                api_username = user_valves.username or self.valves.username
                api_key = user_valves.api_key or self.valves.api_key
                split_query = user_valves.split_query
            else:
                api_username = self.valves.username
                api_key = self.valves.api_key
                api_key_auth = True
                split_query = True

            if (api_key_auth and not api_username) or not api_key:
                await event_emitter.emit_status(
                    "Please provide a username and API key or personal access token.",
                    True,
                    True,
                )
                return "Error: Please provide a username and API key or personal access token."

            # Apply memory settings from valves
            global MAX_ISSUE_SIZE, BATCH_SIZE
            MAX_ISSUE_SIZE = self.valves.max_issue_size
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

            # Initialize document retriever and load model
            try:
                if not self.document_retriever:
                    self.document_retriever = JiraDocumentRetriever(
                        model_cache_dir=model_cache_dir,
                        device="cpu" if self.valves.cpu_only else "cuda",
                        embedding_model_name=self.valves.embedding_model_name,
                        batch_size=BATCH_SIZE,
                    )

                if not self.document_retriever.embedding_model:
                    await self.document_retriever.load_embedding_model(event_emitter)
            except Exception as e:
                await event_emitter.emit_status(
                    f"Error loading embedding model: {str(e)}", True, True
                )
                return f"Error: Failed to load embedding model: {str(e)}"

            jira = Jira(
                api_username,
                api_key,
                self.valves.base_url,
                api_key_auth,
                self.valves.ssl_verify,
            )

            await event_emitter.emit_status(
                f"Searching for '{query}' on Jira...", False
            )

            try:
                # Use the new JQL endpoint instead of the deprecated one
                endpoint = "search/jql"
                params = {
                    "jql": query,
                    "maxResults": self.valves.result_limit,
                    "fields": [
                        "summary",
                        "description",
                        "status",
                        "issuetype",
                        "priority",
                        "assignee",
                        "created",
                    ],
                }

                # Make the request to the new JQL endpoint
                raw_response = jira.get(endpoint, params)

                # Extract issues from the response
                if "issues" not in raw_response or not raw_response["issues"]:
                    await event_emitter.emit_status(
                        f"No Jira issues found matching your query: '{query}'", True
                    )
                    return json.dumps([])

                issue_keys = [issue["key"] for issue in raw_response["issues"]]

                if not issue_keys:
                    await event_emitter.emit_status(
                        f"No Jira issues found matching your query: '{query}'", True
                    )
                    return json.dumps([])

                # Fetch detailed information for each issue
                raw_documents = []
                for i, key in enumerate(issue_keys):
                    await event_emitter.emit_status(
                        f"Retrieving Jira issue {i+1}/{len(issue_keys)}...", False
                    )
                    issue = jira.get_issue(key)

                    # Create a document with issue content
                    content = f"Title: {issue['summary']}\n\nDescription: {issue['description']}"
                    raw_documents.append(
                        Document(
                            issue_content=content,
                            metadata={
                                "key": issue["key"],
                                "summary": issue["summary"],
                                "status": issue["status"],
                                "issuetype": issue["issuetype"],
                                "priority": issue["priority"],
                                "assignee": issue["assignee"],
                                "created": issue["created"],
                                "link": issue["link"],
                            },
                        )
                    )

                # If full context mode is enabled, skip RAG processing and return complete issues
                if self.valves.full_context:
                    await event_emitter.emit_status(
                        f"Preparing all {len(raw_documents)} Jira issues for you...",
                        False,
                    )

                    # Create results with the complete content
                    results = []
                    for doc in raw_documents:
                        result = {
                            "key": doc.metadata["key"],
                            "summary": doc.metadata["summary"],
                            "description": doc.issue_content,
                            "status": doc.metadata["status"],
                            "issuetype": doc.metadata["issuetype"],
                            "priority": doc.metadata["priority"],
                            "assignee": doc.metadata["assignee"],
                            "created": doc.metadata["created"],
                            "link": doc.metadata["link"],
                        }

                        # Add citations for each full document
                        await event_emitter.emit_source(
                            result["summary"], result["link"], result["description"]
                        )

                        results.append(result)

                    await event_emitter.emit_status(
                        f"Found {len(results)} complete Jira issues matching '{query}'",
                        True,
                    )

                    return json.dumps(results)

                # Apply RAG processing to find the most relevant content
                elif raw_documents:
                    await event_emitter.emit_status(
                        f"Analyzing {len(raw_documents)} issues to find the most relevant content...",
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
                        await self.document_retriever.retrieve_from_jira_issues(
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

                    # Group chunks by issue to build results
                    issue_chunks = {}
                    for chunk in relevant_chunks:
                        issue_key = chunk.metadata["key"]
                        if issue_key not in issue_chunks:
                            issue_chunks[issue_key] = {
                                "key": issue_key,
                                "summary": chunk.metadata["summary"],
                                "status": chunk.metadata["status"],
                                "issuetype": chunk.metadata["issuetype"],
                                "priority": chunk.metadata["priority"],
                                "assignee": chunk.metadata["assignee"],
                                "created": chunk.metadata["created"],
                                "link": chunk.metadata["link"],
                                "chunks": [],
                            }
                        issue_chunks[issue_key]["chunks"].append(chunk.issue_content)

                    # Create final results
                    results = []
                    for issue_key, issue_data in issue_chunks.items():
                        # Join chunks with context into a coherent description
                        description = "\n\n".join(issue_data["chunks"])

                        result = {
                            "key": issue_data["key"],
                            "summary": issue_data["summary"],
                            "description": description,
                            "status": issue_data["status"],
                            "issuetype": issue_data["issuetype"],
                            "priority": issue_data["priority"],
                            "assignee": issue_data["assignee"],
                            "created": issue_data["created"],
                            "link": issue_data["link"],
                        }

                        # Add citations for each relevant chunk
                        await event_emitter.emit_source(
                            result["summary"], result["link"], result["description"]
                        )

                        results.append(result)
                else:
                    results = []

                # Final check to ensure we have results
                if not results:
                    await event_emitter.emit_status(
                        f"No relevant content found in Jira issues for '{query}'", True
                    )
                    return json.dumps([])

                await event_emitter.emit_status(
                    f"Search complete! Found {len(results)} issues with information about '{query}'",
                    True,
                )

                return json.dumps(results)

            except Exception as e:
                await event_emitter.emit_status(
                    f"Failed to search for '{query}' on Jira: {e}", True, True
                )
                return f"Error: {e}"
        except Exception as e:
            await event_emitter.emit_status(
                f"Unexpected error during search: {str(e)}.", True, True
            )
            return f"Error: {str(e)}"
