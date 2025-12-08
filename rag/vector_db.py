"""
Vector Database module for Legal RAG system using ChromaDB with SentenceTransformer embeddings.
"""

from typing import Any, Dict, List
import chromadb
import torch
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LegalVectorDB:
    """Manages vector database for legal document retrieval using ChromaDB."""
    
    def __init__(
        self, 
        db_path: str = CHROMA_DB_PATH, 
        collection_name: str = COLLECTION_NAME
    ):
        """
        Initialize the Legal Vector Database.
        
        Args:
            db_path: Path to persist the ChromaDB database
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            device=DEVICE
        )
        print(f"Using device: {DEVICE}")
        
        # Initialize or get collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            existing_count = self.collection.count()
            print(
                f"✓ Collection '{self.collection_name}' loaded with "
                f"{existing_count} documents."
            )
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal documents embeddings"}
            )
            print(f"✓ Created new collection '{self.collection_name}'.")
    
    def is_indexed(self) -> bool:
        """
        Check if the collection contains any documents.
        
        Returns:
            True if collection has documents, False otherwise
        """
        try:
            count = self.collection.count()
            return count > 0
        except Exception:
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant legal documents.
        
        Args:
            query: Query string
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()
            
            # Perform similarity search
            search_kwargs = {
                "query_embeddings": query_embedding,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if filter_metadata:
                search_kwargs["where"] = filter_metadata
            
            results = self.collection.query(**search_kwargs)
            
            # Format results
            retrieved_docs = []
            for i in range(len(results["documents"][0])):
                retrieved_docs.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]
                })
            
            return retrieved_docs
        
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database stats
        """
        return {
            "document_count": self.get_document_count(),
            "collection_name": self.collection_name,
            "db_path": self.db_path,
            "is_indexed": self.is_indexed(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "device": str(DEVICE)
        }
    
    def delete_collection(self):
        """Delete the current collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error deleting collection: {e}")