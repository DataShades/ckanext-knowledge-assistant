# encoding: utf-8

"""CLI commands for knowledge assistant management."""

import logging
import click

from ckanext.knowledge_assistant.rag import get_rag_engine

log = logging.getLogger(__name__)


@click.group()
def knowledge_assistant():
    """Knowledge Assistant management commands."""
    pass


@knowledge_assistant.command()
@click.option("--force", is_flag=True, help="Force re-index all datasets")
def index(force):
    """Index CKAN datasets into the vector store.

    Example:
        ckan knowledge-assistant index
        ckan knowledge-assistant index --force
    """
    click.echo("Starting dataset indexing...")

    try:
        engine = get_rag_engine()
        engine.index_datasets(force_refresh=force)
        click.echo("✓ Indexing completed successfully")
    except Exception as e:
        click.echo(f"✗ Error during indexing: {e}", err=True)
        raise


@knowledge_assistant.command()
@click.option("--query", "-q", help="Query to test (interactive if not provided)")
def test_query(query):
    """Test the RAG engine with a query.

    Example:
        ckan knowledge-assistant test-query
        ckan knowledge-assistant test-query -q "Show me finance datasets"
    """
    from ckanext.knowledge_assistant.rag import get_rag_engine

    # Interactive mode if no query provided
    if not query:
        query = click.prompt("Enter your question")

    click.echo(f"\nTesting query: {query}")

    try:
        engine = get_rag_engine()
        response = engine.query(query)
        click.echo("\nResponse:")
        click.echo(response)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise


def get_commands():
    """Return CLI commands for CKAN."""
    return [knowledge_assistant]
