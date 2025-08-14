# ChromaDB operations
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import Dict, dict , Any, List, Optional
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor #  is a standard library module for running tasks asynchronously using threads or processes.
# allows you to run functions in parallel using multiple threads, which is useful for speeding up I/O-bound operations.
import logging
from core.config import settings

logging = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = chromadb.persistent(
            path = settings.CHROMA_PERSIST_DIRECTORY,
            settings= ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        # initialize the embedding model 
        logging.info(f"loading embeddings from {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.executor = ThreadPoolExecutor(max_workers=4) # Create a thread pool with a maximum of 4 worker threads

        logging.info("VectorStore initialized successfully.")

    def _generate_doc_id(self, content: str, metadata: dict[str, Any]) -> str:
        """Generate unique document ID based on content and metadata"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        filename = metadata.get('filename','unknown')
        chunk_index = metadata.get('chunk_index','0')
        return f"{filename}_{chunk_index}_{content_hash[:8]}"
    
    
    def _generate_embeddings(self , text: List[str]):
        """Generate embeddings for a list of texts using the embedding model."""
        try :
            embeddings = self.embedding_model.encode(text, show_progress_bar=False, convert_to_tensor=True)
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

    async def add_documents(self, documents: List[str], metadatas: List[dict[str, Any]], collection_name : str = "documents") -> None:
        """Add documents to the vector store."""
        try :
            logging.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
            # Generate embeddings asynchronously
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_embeddings,
                documents
            )
            # generating unique doc ids
            doc_ids = [self._generate_doc_id(doc, metadata) for doc, metadata in zip(documents, metadatas)]
            collection = self.client.get_or_create_collection(name=collection_name, metadata = {'description': f'document collection:{collection_name}'})
            
            collection.add(
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=documents,
                ids=doc_ids
            )
            logging.info(f"Successfully added documents to collection '{collection_name}'.")
            return doc_ids
        except Exception as e:
            logging.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise Exception(f"vector store error : {str (e)}")
    
    async def similarity_search(self, query : str , collection_name : str = "documents" , n_results : int = 5 ,filter_metadata: Optional[ Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store."""
        try:
            logging.info(f"Searching for similar documents in collection '{collection_name}'")
            collection = self.client.get_collection(name=collection_name)
            if not collection:
                logging.warning(f"Collection '{collection_name}' not found.")
                return []

            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_embeddings,
                [query]
            )
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                logging.warning(f"Collection '{collection_name}' not found")
                return []
            # Perform similarity search
            results = collection.similarity_search(
                query_embeddings=query_embedding,
                n_results=n_results,
                include = ["documents","metadatas","distances"],
                where = filter_metadata
            )
            # format results 
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity_score": 1 - results['distances'][0][i]
                    })
            logging.info(f"Found {len(formatted_results)} similar documents ")
            return formatted_results
        except Exception as e:
            logging.error(f"similarity search failed at: {e}")
            raise Exception(f"search error: {str(e)}")
    async def get_collection_stats(self, collection_name: str = "documents") -> Dict[str, Any]:
        """Get statistics for a specific collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return {"name": collection_name, "document_count": count, 'status' : 'active' if count > 0 else 'empty'}
        except Exception:
            return {"name": collection_name, "document_count": 0, 'status' : 'not_found'}

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a specific collection."""
        try:
            self.client.delete_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' deleted successfully.")
            return True
        except Exception as e:
            logging.error(f"failed to delete '{collection_name}': {e}")
            return False
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the vector store."""
        try:
            collections = self.client.list_collections()
            return [{"name": col.name} for col in collections]
        except Exception as e:
            logging.error(f"failed to list collections: {e}")
            return []

vector_store = VectorStore()