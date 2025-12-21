"""Configuration settings for the Knowledge Assistant extension."""

from __future__ import annotations

import ckan.plugins.toolkit as tk

# Config keys
LLM_PROVIDER = "ckanext.knowledge_assistant.llm_provider"
LLM_MODEL = "ckanext.knowledge_assistant.llm_model"
EMBEDDING_PROVIDER = "ckanext.knowledge_assistant.embedding_provider"
EMBEDDING_MODEL = "ckanext.knowledge_assistant.embedding_model"
OLLAMA_BASE_URL = "ckanext.knowledge_assistant.ollama_base_url"
OPENAI_API_KEY = "ckanext.knowledge_assistant.openai_api_key"
ENABLED = "ckanext.knowledge_assistant.enabled"
VECTOR_STORE_URL = "ckanext.knowledge_assistant.vector_store_url"
VECTOR_STORE_TABLE = "ckanext.knowledge_assistant.vector_store_table"
SIMILARITY_TOP_K = "ckanext.knowledge_assistant.similarity_top_k"


def get_llm_provider() -> str:
    """Get the configured LLM provider."""
    return tk.config[LLM_PROVIDER]


def get_llm_model() -> str:
    """Get the configured LLM model name."""
    return tk.config[LLM_MODEL]


def get_embedding_provider() -> str:
    """Get the configured embedding provider."""
    return tk.config[EMBEDDING_PROVIDER]


def get_embedding_model() -> str:
    """Get the configured embedding model name."""
    return tk.config[EMBEDDING_MODEL]



def get_ollama_base_url() -> str:
    """Get the Ollama server base URL."""
    return tk.config[OLLAMA_BASE_URL]


def get_openai_api_key() -> str:
    """Get the OpenAI API key if configured."""
    return tk.config.get(OPENAI_API_KEY, "")


def is_enabled() -> bool:
    """Check if the knowledge assistant is enabled."""
    return tk.config[ENABLED]


def get_vector_store_connection_string() -> str:
    """Get PostgreSQL connection string for vector store."""
    return tk.config.get(VECTOR_STORE_URL) or tk.config["sqlalchemy.url"]


def get_vector_store_table_name() -> str:
    """Get the table name for vector storage."""
    return tk.config[VECTOR_STORE_TABLE]


def get_similarity_top_k() -> int:
    """Get similarity top k for queries."""
    return tk.config[SIMILARITY_TOP_K]
