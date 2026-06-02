"""PostgreSQL with pgvector connector for VectorDB Benchmarker."""

import os
from typing import List, Dict, Any, Optional
import json

from .base import BaseVectorDB, SearchResult
from .registry import register_vectordb


@register_vectordb("pgvector")
class PgVectorConnector(BaseVectorDB):
    """PostgreSQL with pgvector extension connector.
    
    pgvector adds vector similarity search to PostgreSQL.
    
    Config options:
        - host: PostgreSQL host (default: "localhost")
        - port: PostgreSQL port (default: 5432)
        - database: Database name (default: "vectordb")
        - user: Username (default: "postgres")
        - password: Password (or PGPASSWORD env var)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", os.getenv("PGHOST", "localhost"))
        self.port = config.get("port", int(os.getenv("PGPORT", "5432")))
        self.database = config.get("database", os.getenv("PGDATABASE", "vectordb"))
        self.user = config.get("user", os.getenv("PGUSER", "postgres"))
        self.password = config.get("password", os.getenv("PGPASSWORD", ""))
        self.conn = None
        self._dimensions: Dict[str, int] = {}
    
    def connect(self) -> None:
        """Connect to PostgreSQL."""
        import psycopg2
        
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        
        # Ensure pgvector extension is installed
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
        
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.conn:
            self.conn.close()
        self.conn = None
        self._dimensions.clear()
        self._connected = False
    
    def _table_name(self, collection: str) -> str:
        """Get safe table name."""
        # Replace invalid characters
        return f"vec_{collection.replace('-', '_').replace('.', '_')}"
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a table for vector storage."""
        table = self._table_name(name)
        
        with self.conn.cursor() as cur:
            # Drop if exists
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Create table with vector column
            cur.execute(f"""
                CREATE TABLE {table} (
                    id TEXT PRIMARY KEY,
                    embedding vector({dimension}),
                    document TEXT,
                    metadata JSONB
                )
            """)
            
            # Create index for fast similarity search
            cur.execute(f"""
                CREATE INDEX ON {table} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            self.conn.commit()
        
        self._dimensions[name] = dimension
    
    def delete_collection(self, name: str) -> None:
        """Delete the collection table."""
        try:
            table = self._table_name(name)
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
                self.conn.commit()
            self._dimensions.pop(name, None)
        except Exception:
            pass
    
    def add_documents(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to PostgreSQL."""
        table = self._table_name(collection)
        
        with self.conn.cursor() as cur:
            for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
                doc = documents[i] if documents and i < len(documents) else None
                meta = json.dumps(metadatas[i]) if metadatas and i < len(metadatas) else None
                
                # Convert embedding to pgvector format
                vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
                
                cur.execute(f"""
                    INSERT INTO {table} (id, embedding, document, metadata)
                    VALUES (%s, %s::vector, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        document = EXCLUDED.document,
                        metadata = EXCLUDED.metadata
                """, (doc_id, vec_str, doc, meta))
            
            self.conn.commit()
    
    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        table = self._table_name(collection)
        vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        with self.conn.cursor() as cur:
            # Build query with optional filters
            where_clause = ""
            params = [vec_str, vec_str, top_k]
            
            if filters:
                filter_conditions = []
                for k, v in filters.items():
                    filter_conditions.append(f"metadata->>'{k}' = %s")
                    params.insert(-1, str(v))
                where_clause = "WHERE " + " AND ".join(filter_conditions)
            
            cur.execute(f"""
                SELECT id, document, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM {table}
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params)
            
            results = cur.fetchall()
        
        search_results = []
        for row in results:
            doc_id, document, metadata, score = row
            meta = json.loads(metadata) if metadata else {}
            
            search_results.append(SearchResult(
                id=doc_id,
                score=score,
                document=document,
                metadata=meta,
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        table = self._table_name(collection)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]
