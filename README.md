# VectorDB Benchmarker

An extensible benchmarking tool for evaluating Vector Databases and RAG systems.

## Features

- **Extensible Architecture**: Easy to add new vector databases, embedding providers, and data sources
- **Arabic-First**: Built-in Arabic corpus for voice agents and AI evaluation
- **Comprehensive Metrics**: Latency, memory usage, and retrieval quality (Recall, Precision, MRR, NDCG)
- **Multiple Vector DBs**: ChromaDB, FAISS, Pinecone (extensible to more)
- **Multiple Embedding Providers**: Google Gemini, OpenAI, Sentence Transformers
- **CLI & Programmatic API**: Use from command line or integrate into your code

## Installation

```bash

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# Run benchmark with default settings (ChromaDB, FAISS with Gemini embeddings)
python -m arabic_rag_bench.cli run

# Specify databases and embedding provider
python -m arabic_rag_bench.cli run -d chromadb faiss pinecone -e gemini

# List available components
python -m arabic_rag_bench.cli list databases
python -m arabic_rag_bench.cli list embeddings
```

### Programmatic API

```python
from arabic_rag_bench import Benchmark, BenchmarkConfig

# Quick run with defaults
benchmark = Benchmark()
result = benchmark.run(databases=["chromadb", "faiss"])
result.print_summary()
```

## Configuration

Create a `.env` file with your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key  
PINECONE_API_KEY=your_pinecone_api_key
```

## Project Structure

```
src/arabic_rag_bench/
├── benchmark.py         # Main benchmark engine
├── config.py            # Configuration classes
├── cli.py               # Command-line interface
├── vectordb/            # Vector database connectors
├── embeddings/          # Embedding providers
├── data/                # Arabic corpus and loaders
├── metrics/             # Performance and retrieval metrics
└── rag/                 # RAG evaluation (future)
```

## Extending

### Add a Vector Database

```python
from arabic_rag_bench.vectordb import BaseVectorDB, register_vectordb

@register_vectordb("mydb")
class MyDBConnector(BaseVectorDB):
    # Implement abstract methods...
```

### Add an Embedding Provider

```python
from arabic_rag_bench.embeddings import BaseEmbedding, register_embedding

@register_embedding("myembedding")  
class MyEmbedding(BaseEmbedding):
    # Implement abstract methods...
```

## License

MIT License
