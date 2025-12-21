[![Tests](https://github.com/DataShades/ckanext-knowledge-assistant/workflows/Tests/badge.svg?branch=main)](https://github.com/DataShades/ckanext-knowledge-assistant/actions)

# ckanext-knowledge-assistant

RAG-based knowledge assistant for CKAN - semantic search over datasets using natural language queries.


## Requirements

- CKAN >= 2.10
- Python >= 3.10
- PostgreSQL with pgvector extension
- Ollama (for local models) OR OpenAI API key

Compatibility with core CKAN versions:

| CKAN version    | Compatible?   |
| --------------- | ------------- |
| 2.9 and earlier | not tested    |
| 2.10            | not tested    |
| 2.11            | yes           |

Suggested values:

* "yes"
* "not tested" - I can't think of a reason why it wouldn't work
* "not yet" - there is an intention to get it working
* "no"


## Installation

### 1. Install pgvector
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-15-pgvector

# Or from source
cd /tmp
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Enable pgvector in your database
```bash
sudo -u postgres psql -d your_ckan_db -c "CREATE EXTENSION vector;"
```

### 3. Install Ollama (for local models)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen3:8b-q4_K_M
ollama pull nomic-embed-text
```

### 4. Install ckanext-knowledge-assistant

Activate your CKAN virtual environment:
```bash
. /usr/lib/ckan/default/bin/activate
```

Clone the source and install it on the virtualenv:
```bash
git clone https://github.com/DataShades/ckanext-knowledge-assistant.git
cd ckanext-knowledge-assistant
pip install -e .
pip install -r requirements.txt
```

### 5. Configure CKAN

Add `knowledge_assistant` to the `ckan.plugins` setting in your CKAN config file (by default the config file is located at `/etc/ckan/default/ckan.ini`).

Add the following configuration settings:
```ini
# LLM Configuration
ckanext.knowledge_assistant.llm_provider = ollama
ckanext.knowledge_assistant.llm_model = qwen3:8b-q4_K_M
ckanext.knowledge_assistant.ollama_base_url = http://localhost:11434

# Embedding Configuration
ckanext.knowledge_assistant.embedding_provider = ollama
ckanext.knowledge_assistant.embedding_model = nomic-embed-text

# Vector Store (optional - defaults to CKAN's database)
# ckanext.knowledge_assistant.vector_store_url = postgresql://user:pass@localhost/ckan
ckanext.knowledge_assistant.vector_store_table = knowledge_assistant_embeddings

# Query Settings
ckanext.knowledge_assistant.similarity_top_k = 5
```

### 6. Restart CKAN

For example if you've deployed CKAN with Apache on Ubuntu:
```bash
sudo service apache2 reload
```

### 7. Index your datasets
```bash
ckan -c /etc/ckan/default/ckan.ini knowledge-assistant index
```


## Usage

### CLI Commands

**Index datasets:**
```bash
# Initial indexing
ckan knowledge-assistant index

# Re-index (clears existing data)
ckan knowledge-assistant index --force
```

**Test queries:**
```bash
# Interactive mode
ckan knowledge-assistant test-query

# Direct query
ckan knowledge-assistant test-query -q "Show me datasets about soil"
```

## Developer installation

To install ckanext-knowledge-assistant for development, activate your CKAN virtualenv and
do:

    git clone https://github.com/DataShades/ckanext-knowledge-assistant.git
    cd ckanext-knowledge-assistant
    pip install -e .
    pip install -r dev-requirements.txt


## Tests

To run the tests, do:

    pytest --ckan-ini=test.ini


## Releasing a new version of ckanext-knowledge-assistant

If ckanext-knowledge-assistant should be available on PyPI you can follow these steps to publish a new version:

1. Update the version number in the `pyproject.toml` file. See [PEP 440](http://legacy.python.org/dev/peps/pep-0440/#public-version-identifiers) for how to choose version numbers.

2. Make sure you have the latest version of necessary packages:

    pip install --upgrade setuptools wheel twine

3. Create a source and binary distributions of the new version:

       python -m build && twine check dist/*

   Fix any errors you get.

4. Upload the source distribution to PyPI:

       twine upload dist/*

5. Commit any outstanding changes:

       git commit -a
       git push

6. Tag the new release of the project on GitHub with the version number from
   the `setup.py` file. For example if the version number in `setup.py` is
   0.0.1 then do:

       git tag 0.0.1
       git push --tags

## License

[AGPL](https://www.gnu.org/licenses/agpl-3.0.en.html)
