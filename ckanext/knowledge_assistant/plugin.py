import logging

import ckan.plugins as p
import ckan.plugins.toolkit as tk

log = logging.getLogger(__name__)


@tk.blanket.config_declarations
class KnowledgeAssistantPlugin(p.SingletonPlugin):
    """CKAN Knowledge Assistant Plugin

    Provides RAG-based semantic search capabilities for CKAN datasets
    using LlamaIndex and vector stores.
    """
    p.implements(p.IConfigurer)
    p.implements(p.IClick)

    # IConfigurer

    def update_config(self, config_):
        """Update CKAN config with extension settings."""
        tk.add_template_directory(config_, "templates")
        tk.add_public_directory(config_, "public")
        tk.add_resource("assets", "knowledge_assistant")

    # IClick

    def get_commands(self):
        """Return CLI commands for this extension."""
        from ckanext.knowledge_assistant.cli import get_commands
        return get_commands()
