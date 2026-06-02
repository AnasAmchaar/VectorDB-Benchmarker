# VectorDB Benchmarker

An extensible benchmarking tool for evaluating Vector Databases and RAG systems.

## Features

- **Extensible Architecture**: Easy to add new vector databases, embedding providers, and data sources via a decorator-based plugin system
- **Comprehensive Metrics**: Latency (mean, P95, P99), memory usage, and retrieval quality (Recall@K, MRR, NDCG)
- **11 Vector DBs out of the box**: ChromaDB, FAISS, Pinecone, Milvus, Weaviate, Qdrant, Vald, Turbopuffer, pgvector, Redis, Elasticsearch
- **3 Embedding Providers**: Sentence Transformers (local, no API key), Google Gemini, OpenAI
- **Flexible Data**: Synthetic dataset generator for quick testing, or bring your own JSON dataset
- **CLI & Programmatic API**: Use from command line or integrate into your code

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/VectorDB-Benchmarker.git
cd VectorDB-Benchmarker

# Install as a package (recommended)
pip install -e .

# Or install with dev dependencies (pytest, etc.)
pip install -e .[dev]
```

## Quick Start

### Command Line

```bash
# Run benchmark with defaults (ChromaDB + FAISS, sentence-transformers embeddings, synthetic data)
python -m rag_benchmark.cli run

# Specify databases and embedding provider
python -m rag_benchmark.cli run -d chromadb faiss qdrant -e gemini

# Use your own dataset
python -m rag_benchmark.cli run --data-file ./my_dataset.json

# List all registered databases and embedding providers
python -m rag_benchmark.cli list databases
python -m rag_benchmark.cli list embeddings
```

### Programmatic API

```python
from rag_benchmark import Benchmark, BenchmarkConfig

# Quick run with defaults
benchmark = Benchmark()
result = benchmark.run(databases=["chromadb", "faiss"])
result.print_summary()
```

## Configuration

Create a `.env` file with your API keys (only needed for cloud providers):

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Project Structure

```
src/rag_benchmark/
├── benchmark.py         # Main benchmark engine
├── config.py            # Configuration classes
├── cli.py               # Command-line interface
├── vectordb/            # Vector database connectors (plugin system)
│   ├── base.py          # BaseVectorDB abstract class
│   ├── registry.py      # Decorator-based registry
│   ├── chromadb.py      # ChromaDB connector
│   ├── faiss_db.py      # FAISS connector
│   └── ...              # More connectors
├── embeddings/          # Embedding providers (plugin system)
│   ├── base.py          # BaseEmbedding abstract class
│   ├── registry.py      # Decorator-based registry
│   └── ...              # Provider implementations
├── data/                # Dataset loaders (synthetic, local JSON)
└── metrics/             # Performance and retrieval metrics
```

## Extending

### Add a New Vector Database

Create a new file (e.g., `vectordb/my_db.py`) and use the `@register_vectordb` decorator:

```python
from rag_benchmark.vectordb import BaseVectorDB, register_vectordb, SearchResult

@register_vectordb("mydb")
class MyDBConnector(BaseVectorDB):
    def connect(self) -> None:
        # Connect to your database
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def create_collection(self, name: str, dimension: int) -> None:
        # Create a collection/index
        ...

    def delete_collection(self, name: str) -> None:
        # Delete a collection/index
        ...

    def add_documents(self, collection, ids, embeddings, documents=None, metadatas=None):
        # Insert vectors
        ...

    def search(self, collection, query_embedding, top_k=5, filters=None):
        # Return List[SearchResult]
        ...

    def count(self, collection: str) -> int:
        # Return document count
        ...
```

Then import it in `vectordb/__init__.py`:

```python
from . import my_db  # noqa: F401
```

It will now appear in `list databases` and can be passed to `run -d mydb`.

### Add a New Embedding Provider

Create a new file (e.g., `embeddings/my_embed.py`) and use the `@register_embedding` decorator:

```python
from rag_benchmark.embeddings import BaseEmbedding, register_embedding

@register_embedding("myembedding")
class MyEmbedding(BaseEmbedding):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Return embeddings for a batch of documents
        ...

    def embed_query(self, text: str) -> list[float]:
        # Return embedding for a single query
        ...
```

Then import it in `embeddings/__init__.py`:

```python
from . import my_embed  # noqa: F401
```

### Custom Dataset Format

Pass a JSON file with this structure:

```json
{
  "documents": [
    {"id": "doc_0", "text": "Your document text...", "metadata": {"key": "value"}},
    ...
  ],
  "queries": [
    {"id": "query_0", "text": "Your query text...", "metadata": {}},
    ...
  ],
  "relevance": {
    "0": [0, 3, 7],
    "1": [1, 5]
  }
}
```

Where `relevance` maps query index → list of relevant document indices.

## License

MIT License
