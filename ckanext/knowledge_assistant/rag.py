import re
import logging
from typing import Iterable, Any

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

import ckan.plugins.toolkit as tk
from ckan import model
from ckanext.knowledge_assistant import config

log = logging.getLogger(__name__)

# Global engine instance (initialized on first use)
_rag_engine = None


class RAGEngine:
    """Retrieval-Augmented Generation engine for CKAN dataset search."""

    def __init__(self):
        """Initialize the RAG engine with configured LLM and vector store."""

        self.config = config
        self._setup_llm()
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_index()

    def _setup_llm(self):
        """Configure the LLM based on settings."""
        provider = self.config.get_llm_provider()
        model = self.config.get_llm_model()

        if provider == "ollama":
            base_url = self.config.get_ollama_base_url()
            self.llm = Ollama(model=model, base_url=base_url)
            log.info(f"Using Ollama LLM: {model} at {base_url}")

        elif provider == "openai":
            api_key = self.config.get_openai_api_key()
            self.llm = OpenAI(model=model, api_key=api_key)
            log.info(f"Using OpenAI LLM: {model}")

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        # Set as global default
        Settings.llm = self.llm

    def _setup_embeddings(self):
        """Configure the embedding model based on settings."""
        provider = self.config.get_embedding_provider()
        model_name = self.config.get_embedding_model()

        if provider == "ollama":
            base_url = self.config.get_ollama_base_url()

            self.embed_model = OllamaEmbedding(
                model_name=model_name,
                base_url=base_url,
                embed_batch_size=10
            )
            log.info(f"Using Ollama embeddings: {model_name} at {base_url}")

        elif provider == "openai":
            api_key = self.config.get_openai_api_key()
            self.embed_model = OpenAIEmbedding(
                model=model,
                api_key=api_key
            )
            log.info(f"Using OpenAI embeddings: {model}")

        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

        # Set as global default
        Settings.embed_model = self.embed_model

    def _setup_vector_store(self):
        """Setup PostgreSQL vector store."""
        connection_string = self.config.get_vector_store_connection_string()
        table_name = self.config.get_vector_store_table_name()

        pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
        match = re.match(pattern, connection_string)

        if not match:
            raise ValueError("Invalid connection string format")

        user, password, host, port, database = match.groups()
        self.vector_store = PGVectorStore.from_params(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
            embed_dim=768,
        )
        log.info(f"Connected to vector store: {table_name}")

    def _setup_index(self):
        """Setup or load the vector index."""
        # Try to load existing index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )
        log.info("Loaded vector index from store")

    def query(self, query_text: str) -> str:
        """Query with better retrieval settings."""
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )

        response = query_engine.query(query_text)
        return str(response)

    def index_datasets(self, force_refresh: bool = False):
        """Index CKAN datasets into the vector store.

        Args:
            force_refresh: If True, clear and re-index all datasets
        """
        if force_refresh:
            log.info("Clearing vector store for refresh")
            self._clear_vector_store()

        log.info("Starting dataset indexing...")

        documents = []
        for dataset in _search_packages():
            # Create document from dataset metadata
            text = self._format_dataset_for_indexing(dataset)
            doc = Document(
                doc_id=dataset["id"],
                text=text,
                metadata={
                    "id": dataset["id"],
                    "name": dataset["name"],
                    "title": dataset.get("title", ""),
                    "type": dataset.get("type", "dataset"),
                    "organization": dataset.get("organization", {}).get("name", ""),
                    "num_resources": dataset.get("num_resources", 0),
                    "license_id": dataset.get("license_id", ""),
                    "formats": ",".join([
                            r.get("format", "").upper()
                            for r in dataset.get("resources", [])
                            if r.get("format")
                        ]),
                    "metadata_created": dataset.get("metadata_created", ""),
                    "metadata_modified": dataset.get("metadata_modified", "")
                }
            )
            documents.append(doc)

        if documents:
            self.index.insert_nodes(documents)
            log.info(f"Indexed {len(documents)} datasets")
        else:
            log.warning("No datasets found to index")

    def clean_markdown(self, text: str) -> str:
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # links
        text = re.sub(r'#+\s*', '', text)  # headings
        return text

    def _format_dataset_for_indexing(self, dataset: dict) -> str:
        """Format only semantically relevant fields."""
        parts = []

        # Primary searchable content
        parts.append(f"Title: {dataset.get('title', '')}")

        notes = dataset.get('notes', '')[:8000]
        if notes:
            # Strip markdown for cleaner embedding
            notes = self.clean_markdown(notes)
            parts.append(f"Description: {notes}")

        org = dataset.get('organization', {})
        if org:
            parts.append(f"Organization: {org.get('title', '')}")

        tags = [t['name'] for t in dataset.get('tags', [])]
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        if dataset.get('author'):
            parts.append(f"Author: {dataset['author']}")

        for field in ['purpose', 'lineage', 'spatial_coverage']:
            if dataset.get(field):
                parts.append(f"{field.replace('_', ' ').title()}: {str(dataset[field])[:500]}")

        resources = dataset.get('resources', [])
        if resources:
            formats = [r.get('format', '').upper() for r in resources if r.get('format')]
            formats = list(set(formats))
            if formats:
                parts.append(f"Available formats: {', '.join(formats)}")
            parts.append(f"Contains {len(resources)} resource(s)")

        return "\n".join(parts)

    def _clear_vector_store(self):
        """Clear all data from the vector store table."""
        from sqlalchemy import create_engine, text

        connection_string = self.config.get_vector_store_connection_string()
        table_name = self.config.get_vector_store_table_name()
        actual_table_name = f"data_{table_name}"

        engine = create_engine(connection_string)
        try:
            with engine.connect() as conn:
                trans = conn.begin()
                conn.execute(text(f"TRUNCATE TABLE {actual_table_name}"))
                trans.commit()
            log.info(f"Cleared vector store table: {actual_table_name}")
        except Exception as e:
            log.warning(f"Could not TRUNCATE, trying DELETE: {e}")
            try:
                with engine.connect() as conn:
                    trans = conn.begin()
                    conn.execute(text(f"DELETE FROM {actual_table_name}"))
                    trans.commit()
                log.info("Cleared vector store using DELETE")
            except Exception as e2:
                log.error(f"Failed to clear vector store: {e2}")
                raise


def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine instance.

    Returns:
        Initialized RAG engine
    """
    global _rag_engine
    if _rag_engine is None:
        log.info("Initializing RAG engine")
        _rag_engine = RAGEngine()

    return _rag_engine


def _search_packages(query: str = "", rows: int = 50) -> Iterable[dict[str, Any]]:
    """Generator function to lookup packages by 'query' incrementally."""
    start = 0
    while True:
        record_packages = tk.get_action("package_search")(
            {"ignore_auth": True},
            {
                "q": query,
                "rows": rows,
                "start": start,
            },
        )

        yield from record_packages["results"]

        start += len(record_packages["results"])

        if start >= record_packages["count"]:
            break
