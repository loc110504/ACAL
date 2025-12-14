"""
RAG (Retrieval-Augmented Generation) module for Legal Documents
Using LangChain Chroma with Azure OpenAI embeddings.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm.auto import tqdm
load_dotenv()


class RAGModule:
    """Manages retrieval-augmented generation for legal documents using LangChain Chroma with Azure OpenAI embeddings."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG module.
        
        Args:
            persist_directory: Directory to persist the ChromaDB collection
        """
        self.persist_directory = persist_directory
        
        # Initialize OpenAI embeddings
        self.embeddings = self._create_embeddings()
        
        # Vector store will be initialized when loading or creating collection
        self.vectorstore = None
        self.collection_name = None
    
    def _create_embeddings(self):
        """
        Create Azure OpenAI embeddings client.
        
        Returns:
            AzureOpenAIEmbeddings instance
        """
        # Get Azure OpenAI embedding configuration from environment
        api_key = os.getenv("OPENAI_API")
        
        embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model="text-embedding-3-small"
            )

        
        return embeddings
    
    def create_collection_from_docs(
        self,
        docs_path: str,
        collection_name: str = "legal_hearsay_docs"
    ):
        """
        Create a Chroma collection from legal documents in a folder.
        Show progress with tqdm.
        """
        self.collection_name = collection_name

        # 1) Collect files
        all_files = []
        for root, _, files in os.walk(docs_path):
            for f in files:
                if f.lower().endswith((".txt", ".md", ".pdf")):
                    all_files.append(os.path.join(root, f))

        if not all_files:
            raise FileNotFoundError(f"No documents found in {docs_path}.")

        print(f"📂 Found {len(all_files)} files to load")

        # 2) Load docs with progress
        documents = []
        for file_path in tqdm(all_files, desc="Loading docs"):
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".txt", ".md"]:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    documents.extend(docs)
                elif ext == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")

        print(f"   ✓ Loaded {len(documents)} raw docs")

        # 3) Split docs with progress
        print("📌 Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        split_docs = []
        for doc in tqdm(documents, desc="Splitting docs"):
            split_docs.extend(text_splitter.split_documents([doc]))

        print(f"   ✓ Split into {len(split_docs)} chunks")

        # 4) Ensure metadata
        for i, doc in enumerate(split_docs):
            doc.metadata.setdefault("chunk_id", i)
            doc.metadata.setdefault("source", doc.metadata.get("source", "unknown"))
            doc.metadata.setdefault("doc_type", "legal_document")

        # 5) Create Chroma store
        print("🔄 Creating embeddings and indexing chunks...")

        # Empty vectorstore initially
        self.vectorstore = None

        for idx, doc in enumerate(tqdm(split_docs, desc="Indexing to Chroma")):
            if self.vectorstore is None:
                # create with first doc
                self.vectorstore = Chroma.from_documents(
                    documents=[doc],
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=self.persist_directory
                )
            else:
                # add additional docs one by one
                self.vectorstore.add_documents([doc])

            # persist on interval
            if idx % 100 == 0:
                self.vectorstore.persist()

        # final persist
        self.vectorstore.persist()

        print(f"✓ Successfully added {len(split_docs)} chunks to '{collection_name}'")
        return self.vectorstore    
    
    def load_collection(self, collection_name: str = "legal_hearsay_docs"):
        """
        Load an existing ChromaDB collection.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            The loaded Chroma vectorstore or None if not found
        """
        self.collection_name = collection_name
        
        try:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"✓ Loaded existing collection '{collection_name}'")
            return self.vectorstore
        except Exception as e:
            print(f"⚠️  Collection '{collection_name}' not found: {e}")
            return None
    
    def query_rag(
        self, 
        query: str, 
        top_k: int = 3,
        collection_name: Optional[str] = None
    ):
        """
        Query the RAG system for relevant medical evidence.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            collection_name: Optional collection name to query
            
        Returns:
            List of evidence dictionaries with doc_id, chunk_id, text
        """
        if collection_name:
            self.load_collection(collection_name)
        
        if not self.vectorstore:
            print("Warning: No collection loaded. Returning empty results.")
            return []
        
        # Perform similarity search
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # Format results
            evidences = []
            for i, (doc, score) in enumerate(results):
                evidences.append({
                    "doc_id": f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', i)}",
                    "chunk_id": doc.metadata.get("chunk_id", i),
                    "source": doc.metadata.get("source", "unknown"),
                    "text": doc.page_content,
                    "score": float(score)  # Similarity score
                })
            
            return evidences
        
        except Exception as e:
            print(f"Error querying vectorstore: {e}")
            return []
        
        except Exception as e:
            print(f"❌ Error querying vectorstore: {e}")
            import traceback
            traceback.print_exc()
            return []