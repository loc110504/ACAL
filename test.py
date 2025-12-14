"""
RAG (Retrieval-Augmented Generation) module using LangChain Chroma with Azure OpenAI embeddings.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document

load_dotenv()


class RAGModule:
    """Manages retrieval-augmented generation using LangChain Chroma with Azure OpenAI embeddings."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG module.
        
        Args:
            persist_directory: Directory to persist the ChromaDB collection
        """
        self.persist_directory = persist_directory
        
        # Initialize Azure OpenAI embeddings
        self.embeddings = self._create_azure_embeddings()
        
        # Vector store will be initialized when loading or creating collection
        self.vectorstore = None
        self.collection_name = None
    
    def _create_azure_embeddings(self) -> AzureOpenAIEmbeddings:
        """
        Create Azure OpenAI embeddings client.
        
        Returns:
            AzureOpenAIEmbeddings instance
        """
        # Get Azure OpenAI embedding configuration from environment
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        api_key = os.getenv("AZURE_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large-2")
        
        if not azure_endpoint or not api_key:
            raise ValueError(
                "Missing AZURE_ENDPOINT or AZURE_API_KEY environment variables. "
                "Please set them in your .env file."
            )
        
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=embedding_deployment,
            model="text-embedding-3-large"  # Model name
        )
        
        return embeddings
    
    def create_collection_from_docs(
        self, 
        docs_path: str, 
        collection_name: str = "phq8_medical_docs"
    ):
        """
        Create a Chroma collection from medical documents in a folder.
        
        Args:
            docs_path: Path to folder containing medical documents
            collection_name: Name of the collection to create
            
        Returns:
            The created Chroma vectorstore
        """
        self.collection_name = collection_name
        
        # Load documents
        documents = []
        
        if os.path.exists(docs_path):
            print(f"Loading documents from {docs_path}...")
            
            # Load text and markdown files
            try:
                txt_loader = DirectoryLoader(
                    docs_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                txt_docs = txt_loader.load()
                documents.extend(txt_docs)
                print(f"Loaded {len(txt_docs)} .txt files")
            except Exception as e:
                print(f"Error loading .txt files: {e}")
            
            try:
                md_loader = DirectoryLoader(
                    docs_path,
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                md_docs = md_loader.load()
                documents.extend(md_docs)
                print(f"Loaded {len(md_docs)} .md files")
            except Exception as e:
                print(f"Error loading .md files: {e}")
            
            # For PDFs, you can use PyPDFLoader
            try:
                from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
                pdf_loader = DirectoryLoader(
                    docs_path,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                print(f"Loaded {len(pdf_docs)} .pdf files")
            except ImportError:
                print("PyPDF not installed. Skipping PDF files. Install with: pip install pypdf")
            except Exception as e:
                print(f"Error loading .pdf files: {e}")
        
        # If no documents found, create default knowledge
        if not documents:
            print(f"Warning: No documents found in {docs_path}")
            print("Creating default medical knowledge...")
            knowledge = create_default_medical_knowledge()
            documents = [
                Document(
                    page_content=text,
                    metadata={"source": "default_knowledge", "chunk_id": i}
                )
                for i, text in enumerate(knowledge)
            ]
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        
        # Add metadata
        for i, doc in enumerate(split_docs):
            if "chunk_id" not in doc.metadata:
                doc.metadata["chunk_id"] = i
            if "doc_type" not in doc.metadata:
                doc.metadata["doc_type"] = "medical_guideline"
        
        # Create Chroma vectorstore
        print("Creating embeddings and storing in Chroma...")
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"✓ Added {len(split_docs)} document chunks to collection '{collection_name}'")
        
        return self.vectorstore
    
    def load_collection(self, collection_name: str = "phq8_medical_docs"):
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
            print(f"✓ Loaded collection '{collection_name}'")
            return self.vectorstore
        except Exception as e:
            print(f"Collection '{collection_name}' not found or error loading: {e}")
            return None
    
    def query_rag(
        self, 
        query: str, 
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> List[Dict]:
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
    
    def build_rag_query(
        self, 
        criterion_name: str, 
        symptom_analysis: str, 
        dialogue_snippet: str = ""
    ) -> str:
        """
        Build an optimized query for RAG retrieval.
        
        Args:
            criterion_name: Name of the PHQ-8 criterion
            symptom_analysis: Summary of symptom analysis
            dialogue_snippet: Optional dialogue context
            
        Returns:
            Optimized query string
        """
        query_parts = [
            f"PHQ-8 {criterion_name}",
            symptom_analysis
        ]
        
        if dialogue_snippet:
            query_parts.append(dialogue_snippet[:200])  # Limit snippet length
        
        query = " ".join(query_parts)
        return query


def main():
    # =========================
    # 1. Initialize RAG module
    # =========================
    rag = RAGModule(
        persist_directory="./chroma_db"
    )

    # =========================
    # 2. Create collection
    # =========================
    vectorstore = rag.create_collection_from_docs(
        docs_path="./legal_docs",
        collection_name="phq8_medical_docs"
    )

    print("\nCollection created successfully\n")

    # =========================
    # 3. Load existing collection (optional test)
    # =========================
    rag.load_collection("phq8_medical_docs")

    # =========================
    # 4. Build optimized RAG query
    # =========================
    rag_query = rag.build_rag_query(
        criterion_name="Anhedonia",
        symptom_analysis="Patient reports loss of interest in daily activities for more than 2 weeks",
        dialogue_snippet="I don't feel motivated to do things I used to enjoy"
    )

    print("RAG QUERY:")
    print(rag_query)
    print("-" * 80)

    # =========================
    # 5. Query RAG
    # =========================
    evidences = rag.query_rag(
        query=rag_query,
        top_k=3
    )

    # =========================
    # 6. Print results
    # =========================
    print(f"Retrieved {len(evidences)} evidence chunks\n")

    for idx, ev in enumerate(evidences, start=1):
        print(f"[{idx}] DOC ID : {ev['doc_id']}")
        print(f"    SOURCE : {ev['source']}")
        print(f"    SCORE  : {ev['score']:.4f}")
        print(f"    TEXT   : {ev['text'][:300]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()

