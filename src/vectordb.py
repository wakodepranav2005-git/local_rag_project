import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Max characters per chunk
            chunk_overlap=150, # Characters to overlap between chunks
            separators=["\n\n", "\n", ". ", " ", ""] # How to split
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into smaller chunks for better retrieval using LangChain.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # === IMPLEMENTATION (Step 3) ===
        chunks = self.text_splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents and add them to the vector database.

        Args:
            documents: List of documents with 'content' and 'metadata'
        """
        # === IMPLEMENTATION (Step 4) ===
        print(f"Processing and ingesting {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # 1. Chunk the document
            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                # 2. Create metadata for the chunk
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx
                all_metadatas.append(chunk_metadata)
                
                # 3. Create a unique ID for each chunk
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")

        if not all_chunks:
            print("No text content found in documents to add.")
            return

        # 4. Create embeddings for all chunks in a single batch
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        # 5. Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"Successfully added {len(all_chunks)} chunks to the vector database.")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results
        """
        # === IMPLEMENTATION (Step 5) ===
        
        # 1. Create embedding for the query
        query_embedding = self.embedding_model.encode([query]).tolist()

        # 2. Search the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # 3. Format and return results
        # ChromaDB results are nested in lists (e.g., {'documents': [['doc1', 'doc2']]})
        # We flatten them for easier use.
        formatted_results = {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
            "ids": results.get("ids", [[]])[0],
        }

        return formatted_results