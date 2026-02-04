"""LangChain RAG framework connector.

LangChain is a flexible framework for building LLM-powered applications.
It provides extensive vectorDB integrations and customizable RAG chains.

Supported VectorDBs: ChromaDB, FAISS, Pinecone, Qdrant, Milvus, Weaviate
"""

import os
import time
from typing import List, Dict, Any, Optional

from .base import BaseRAGFramework, RAGConfig, RAGQueryResult, VectorDBType
from .registry import register_framework


@register_framework("langchain")
class LangChainFramework(BaseRAGFramework):
    """LangChain RAG framework implementation.
    
    LangChain offers:
    - Modular chain composition
    - Many vectorDB integrations
    - Flexible document loaders and text splitters
    - Custom retrieval strategies
    """
    
    VECTORDB_MAPPING = {
        VectorDBType.CHROMADB: "Chroma",
        VectorDBType.FAISS: "FAISS",
        VectorDBType.PINECONE: "Pinecone",
        VectorDBType.QDRANT: "Qdrant",
        VectorDBType.MILVUS: "Milvus",
        VectorDBType.WEAVIATE: "Weaviate",
    }
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._vectorstore = None
        self._retriever = None
        self._rag_chain = None
        self._embeddings = None
        self._llm = None
        self._text_splitter = None
        self._doc_count = 0
    
    @property
    def name(self) -> str:
        return "LangChain"
    
    @property
    def supported_vectordbs(self) -> List[VectorDBType]:
        return list(self.VECTORDB_MAPPING.keys())
    
    def _setup_embeddings(self):
        """Set up embedding model."""
        if self.config.embedding_provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=f"models/{self.config.embedding_model}",
                google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            )
        elif self.config.embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.config.embedding_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
    
    def _setup_llm(self):
        """Set up the LLM."""
        if self.config.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            )
        elif self.config.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.config.llm_provider == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=self.config.llm_model,
                base_url=self.config.extra_config.get("ollama_url", "http://localhost:11434"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def _setup_vectorstore(self):
        """Set up the vector store."""
        vdb_type = self.config.vectordb_type
        vdb_config = self.config.vectordb_config
        
        if vdb_type == VectorDBType.CHROMADB:
            from langchain_chroma import Chroma
            
            persist_dir = vdb_config.get("persist_dir", f"{self.config.storage_dir}/chroma")
            self._vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self._embeddings,
                persist_directory=persist_dir,
            )
            
        elif vdb_type == VectorDBType.FAISS:
            from langchain_community.vectorstores import FAISS
            
            # FAISS needs to be initialized with documents
            # We'll create an empty placeholder
            self._vectorstore = None  # Will be created when documents are added
            
        elif vdb_type == VectorDBType.PINECONE:
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone
            
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = vdb_config.get("index_name", self.config.collection_name)
            
            # Create index if needed
            if index_name not in [idx.name for idx in pc.list_indexes()]:
                pc.create_index(
                    name=index_name,
                    dimension=self.config.embedding_dimension,
                    metric="cosine",
                    spec=vdb_config.get("spec", {"serverless": {"cloud": "aws", "region": "us-east-1"}})
                )
            
            self._vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=self._embeddings,
            )
            
        elif vdb_type == VectorDBType.QDRANT:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            
            url = vdb_config.get("url", "http://localhost:6333")
            client = QdrantClient(url=url)
            
            self._vectorstore = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
                embedding=self._embeddings,
            )
            
        elif vdb_type == VectorDBType.MILVUS:
            from langchain_milvus import Milvus
            
            self._vectorstore = Milvus(
                embedding_function=self._embeddings,
                connection_args={"uri": vdb_config.get("uri", "http://localhost:19530")},
                collection_name=self.config.collection_name,
            )
            
        elif vdb_type == VectorDBType.WEAVIATE:
            from langchain_weaviate import WeaviateVectorStore
            import weaviate
            
            client = weaviate.Client(url=vdb_config.get("url", "http://localhost:8080"))
            self._vectorstore = WeaviateVectorStore(
                client=client,
                index_name=self.config.collection_name,
                embedding=self._embeddings,
            )
        else:
            raise ValueError(f"Unsupported vectordb: {vdb_type}")
    
    def _setup_rag_chain(self):
        """Set up the RAG chain."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        # Create retriever
        if self._vectorstore is not None:
            self._retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
        
        # RAG prompt
        template = """أجب على السؤال بناءً على السياق التالي فقط. إذا لم تتمكن من الإجابة من السياق، قل "لا أعرف".

السياق:
{context}

السؤال: {question}

الإجابة:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        if self._retriever is not None:
            self._rag_chain = (
                {"context": self._retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
            )
    
    def initialize(self) -> None:
        """Initialize LangChain with configured components."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Setup text splitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "،", " ", ""],
        )
        
        self._setup_embeddings()
        self._setup_llm()
        self._setup_vectorstore()
        self._setup_rag_chain()
        
        self._initialized = True
        print(f"✓ LangChain initialized with {self.config.vectordb_type.value}")
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add documents to the vector store."""
        from langchain_core.documents import Document
        
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")
        
        # Create documents with metadata
        docs = []
        for i, text in enumerate(documents):
            metadata = metadatas[i].copy() if metadatas else {}
            metadata["doc_id"] = doc_ids[i] if doc_ids else f"doc_{i}"
            docs.append(Document(page_content=text, metadata=metadata))
        
        # Split documents
        split_docs = self._text_splitter.split_documents(docs)
        
        # Handle FAISS special case (needs initial documents)
        if self.config.vectordb_type == VectorDBType.FAISS:
            from langchain_community.vectorstores import FAISS
            
            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(
                    split_docs,
                    self._embeddings,
                )
            else:
                self._vectorstore.add_documents(split_docs)
        else:
            self._vectorstore.add_documents(split_docs)
        
        # Update retriever and chain
        self._retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        self._setup_rag_chain()
        
        self._doc_count += len(documents)
        return len(split_docs)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """Query the RAG system."""
        if not self._initialized or self._rag_chain is None:
            raise RuntimeError("Framework not initialized or no documents added.")
        
        k = top_k or self.config.top_k
        
        # Update retriever if top_k changed
        if k != self.config.top_k and self._vectorstore is not None:
            self._retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            self._setup_rag_chain()
        
        start_time = time.time()
        
        # Get contexts first
        retrieved_docs = self._retriever.invoke(query)
        
        # Generate answer
        answer = self._rag_chain.invoke(query)
        
        latency = (time.time() - start_time) * 1000
        
        contexts = [doc.page_content for doc in retrieved_docs]
        source_docs = [doc.metadata.get("doc_id", "") for doc in retrieved_docs]
        
        return RAGQueryResult(
            query=query,
            answer=answer,
            contexts=contexts,
            source_docs=source_docs,
            scores=[],  # LangChain default retriever doesn't return scores
            latency_ms=latency,
        )
    
    def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """Retrieve contexts without generating an answer."""
        if not self._initialized or self._retriever is None:
            raise RuntimeError("Framework not initialized or no documents added.")
        
        k = top_k or self.config.top_k
        
        if k != self.config.top_k and self._vectorstore is not None:
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            retriever = self._retriever
        
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]
    
    def clear(self) -> None:
        """Clear all documents."""
        self._vectorstore = None
        self._retriever = None
        self._rag_chain = None
        self._doc_count = 0
        
        # Reinitialize
        self._setup_vectorstore()
        self._setup_rag_chain()
    
    def close(self) -> None:
        """Clean up resources."""
        self._vectorstore = None
        self._retriever = None
        self._rag_chain = None
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get framework statistics."""
        stats = super().get_stats()
        stats.update({
            "document_count": self._doc_count,
            "has_vectorstore": self._vectorstore is not None,
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
        })
        return stats
