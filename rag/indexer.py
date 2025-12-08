"""
Indexer module for loading and indexing legal documents from PDF files into vector database.
"""

import os
import uuid
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from vector_db import LegalVectorDB

try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    PDF_SUPPORT = True
except ImportError:
    print("Warning: PyPDF or LangChain not installed. Install with: pip install pypdf langchain langchain-community")
    PDF_SUPPORT = False


class LegalDocumentIndexer:
    """Handles indexing of legal documents from PDF files."""
    
    def __init__(
        self,
        docs_path: str,
        vector_db: LegalVectorDB = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the document indexer.
        
        Args:
            docs_path: Path to folder containing PDF documents
            vector_db: LegalVectorDB instance (creates new if None)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_path = docs_path
        self.vector_db = vector_db or LegalVectorDB()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf_documents(self) -> List[Dict]:
        """
        Load all PDF documents from the specified directory.
        
        Returns:
            List of document dictionaries with content and metadata
        """
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF support not available. Install: pip install pypdf langchain langchain-community"
            )
        
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents path not found: {self.docs_path}")
        
        print(f"Loading PDF documents from {self.docs_path}...")
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                self.docs_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = pdf_loader.load()
            print(f"✓ Loaded {len(documents)} pages from PDF files")
            
            return documents
        
        except Exception as e:
            print(f"Error loading PDF files: {e}")
            return []
    
    def split_documents(self, documents: List) -> List[Dict]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of LangChain documents
            
        Returns:
            List of chunked documents with metadata
        """
        print("Splitting documents into chunks...")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk_data = {
                "text": doc.page_content,
                "metadata": {
                    "chunk_id": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "chunk_length": len(doc.page_content)
                }
            }
            chunks.append(chunk_data)
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def index_documents(
        self, 
        batch_size: int = 100,
        force_reindex: bool = False
    ) -> bool:
        """
        Index legal documents into the vector database.
        
        Args:
            batch_size: Number of documents to process in each batch
            force_reindex: If True, delete existing collection and re-index
            
        Returns:
            True if successful, False otherwise
        """
        print("=" * 60)
        print("Starting Legal Documents Indexing Process")
        print("=" * 60)
        
        # Check if already indexed
        if self.vector_db.is_indexed() and not force_reindex:
            doc_count = self.vector_db.get_document_count()
            print(f"Database already contains {doc_count} documents.")
            reindex = input("Do you want to re-index anyway? (y/n): ").lower() == "y"
            if not reindex:
                print("Using existing database.")
                return True
            force_reindex = True
        
        # Delete existing collection if re-indexing
        if force_reindex and self.vector_db.is_indexed():
            print("Deleting existing collection...")
            self.vector_db.delete_collection()
            # Re-initialize collection
            self.vector_db._initialize_collection()
        
        try:
            # Load PDF documents
            documents = self.load_pdf_documents()
            
            if not documents:
                print("No documents found to index.")
                return False
            
            # Split into chunks
            chunks = self.split_documents(documents)
            
            if not chunks:
                print("No chunks created from documents.")
                return False
            
            # Index in batches
            print("\nStarting indexing process...")
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            indexed_count = 0
            skipped_count = 0
            
            for batch_idx in tqdm(range(total_batches), desc="Indexing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                batch_chunks = chunks[start_idx:end_idx]
                
                documents_batch = []
                metadatas_batch = []
                ids_batch = []
                
                for chunk in batch_chunks:
                    text = chunk["text"].strip()
                    
                    if not text or len(text) < 10:
                        skipped_count += 1
                        continue
                    
                    documents_batch.append(text)
                    metadatas_batch.append(chunk["metadata"])
                    ids_batch.append(str(uuid.uuid4()))
                
                if not documents_batch:
                    continue
                
                # Generate embeddings
                try:
                    embeddings = self.vector_db.embedding_model.encode(
                        documents_batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    ).tolist()
                    
                    # Add to ChromaDB
                    self.vector_db.collection.add(
                        embeddings=embeddings,
                        documents=documents_batch,
                        metadatas=metadatas_batch,
                        ids=ids_batch
                    )
                    
                    indexed_count += len(documents_batch)
                
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {e}")
                    continue
            
            # Final statistics
            final_count = self.vector_db.get_document_count()
            print("\n" + "=" * 60)
            print("Indexing Completed!")
            print("=" * 60)
            print(f"Total chunks indexed: {final_count}")
            print(f"Chunks processed: {indexed_count}")
            print(f"Chunks skipped: {skipped_count}")
            print("=" * 60)
            
            return True
        
        except Exception as e:
            print(f"Error during indexing: {e}")
            return False


def index_legal_documents(
    docs_path: str = "./legal_documents",
    force_reindex: bool = False
):
    """
    Main function to index legal documents.
    
    Args:
        docs_path: Path to folder containing PDF documents
        force_reindex: If True, delete existing collection and re-index
    """
    # Initialize vector DB
    vector_db = LegalVectorDB()
    
    # Create indexer
    indexer = LegalDocumentIndexer(
        docs_path=docs_path,
        vector_db=vector_db
    )
    
    # Index documents
    success = indexer.index_documents(force_reindex=force_reindex)
    
    if success:
        stats = vector_db.get_stats()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return vector_db


if __name__ == "__main__":
    """Run indexing standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index legal documents from PDF files")
    parser.add_argument(
        "--docs_path", 
        type=str, 
        default="./legal_documents",
        help="Path to folder containing PDF documents"
    )
    parser.add_argument(
        "--force_reindex",
        action="store_true",
        help="Force re-indexing even if database exists"
    )
    
    args = parser.parse_args()
    
    index_legal_documents(
        docs_path=args.docs_path,
        force_reindex=args.force_reindex
    )