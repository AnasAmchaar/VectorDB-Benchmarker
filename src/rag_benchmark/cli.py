"""Command-line interface for VectorDB Benchmarker."""

import argparse
import sys
from .benchmark import Benchmark, BenchmarkConfig
from .vectordb import VectorDBRegistry
from .embeddings import EmbeddingRegistry


def run_benchmark(args):
    print("=" * 70)
    print("VectorDB Benchmarker")
    print("=" * 70)

    config = BenchmarkConfig.default()
    if args.embeddings:
        config.embedding.provider = args.embeddings

    # Simple defaults for demo
    if config.embedding.provider == "gemini":
        config.embedding.model = "text-embedding-004"
    elif config.embedding.provider == "sentence-transformers":
        config.embedding.model = "paraphrase-multilingual-MiniLM-L12-v2"

    if args.data_source:
        config.data.source = args.data_source
    if args.data_file:
        config.data.source = "local"
        config.data.file_path = args.data_file

    benchmark = Benchmark(config)

    dbs_to_run = args.databases if args.databases else ["chromadb", "faiss"]

    result = benchmark.run(databases=dbs_to_run)
    result.print_summary()
    benchmark.save_results(result)


def list_components(args):
    if args.component == "databases":
        available = VectorDBRegistry.list_available()
        print(f"Registered Vector Databases ({len(available)}):")
        for db in available:
            print(f"  - {db}")
    elif args.component == "embeddings":
        available = EmbeddingRegistry.list_available()
        print(f"Registered Embedding Providers ({len(available)}):")
        for emb in available:
            print(f"  - {emb}")
    else:
        print(f"Unknown component: {args.component}")


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB Benchmarker — benchmark vector databases with ease.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s run                                    Run with defaults (ChromaDB + FAISS, sentence-transformers)
  %(prog)s run -d chromadb faiss qdrant            Benchmark specific databases
  %(prog)s run -e gemini                           Use Gemini embeddings
  %(prog)s run --data-file ./my_dataset.json       Use a custom dataset
  %(prog)s list databases                          Show all registered databases
  %(prog)s list embeddings                         Show all registered embedding providers
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument(
        "-d", "--databases", nargs="+",
        help="Databases to benchmark (any registered name). Use 'list databases' to see options.",
    )
    run_parser.add_argument(
        "-e", "--embeddings",
        help="Embedding provider (any registered name). Use 'list embeddings' to see options.",
        default="sentence-transformers",
    )
    run_parser.add_argument(
        "--data-source",
        help="Data source type: 'synthetic' (default) or 'local'.",
        default=None,
    )
    run_parser.add_argument(
        "--data-file",
        help="Path to a local JSON dataset file (implies --data-source local).",
        default=None,
    )
    run_parser.set_defaults(func=run_benchmark)

    # List command
    list_parser = subparsers.add_parser("list", help="List supported components")
    list_parser.add_argument(
        "component", choices=["databases", "embeddings"],
        help="Component type to list",
    )
    list_parser.set_defaults(func=list_components)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
