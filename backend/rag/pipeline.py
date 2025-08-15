# Main RAG implementation
from typing import List, Dict, Any, AsyncGenerator, Optional
import logging
from datetime import datetime
import hashlib

from embeddings.vector_store import VectorStore
from llm.providers import LLMProvider
from prompts.level_prompts import ExplanationLevel,get_prompt_for_level
from utils.text_processing import chunk_text,extract_text_from_file
from utils.cache import cache_manager
from storage.file_manager import file_manager


logger = logging.getLogger(__name__) 

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_provider = LLMProvider()
        self.cache_manager = cache_manager
        self.file_manager = file_manager
        logger.info("RAGPipeline initialized")

    async def process_document(self, file_content: bytes, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        try : 
            logger.info(f"Processing document: {filename}")
            # step 1 : save file    
            file_info = await self.file_manager.save_file(file_content, filename)
            document_id = file_info['saved_filename']

            # step 2 : extract text
            extraction_results = extract_text_from_file( file_content,filename)
            text_content = extraction_results['text']
            metadata = extraction_results['metadata']
            
            # step 3 : chunk the text 
            chunks = chunk_text(text_content, chunk_size, chunk_overlap)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # step 4 : prepare metadata for each chunk 
            chunk_metadatas = []
            for i , chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    'document_id': document_id,
                    'chunk_index': i,
                    'chunk_count': len(chunks),
                    'processing_time': datetime.now().isoformat(),
                }
                chunk_metadatas.append(chunk_metadata)
            
            # step 5 : add to vector store
            docs_ids = await self.vector_store.add_documents(documents=chunks,metadatas=chunk_metadatas,collection_name="documents")

            # step 6 : cache processing status
            processing_results = {
                'document_id': document_id,
                'original_filename': filename,
                'chunk_created': len(chunks),
                'total_words': metadata['word_count'],
                'total_chars': metadata['char_count'],
                'file_type': metadata['file_type'],
                'processing_status': 'completed',
                'chunk_ids': docs_ids
            }
            await self.cache_manager.set(self.cache_manager.document_key(document_id,"processing"),processing_results,ttl=86400)
            logger.info(f"Document {filename} processed successfully")
            return processing_results
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise Exception(f"Failed to process document: {str(e)}")
async def query(
        self,
        question: str,
        level: ExplanationLevel,
        document_id: Optional[str] = None,
        max_context_length: int = 3000,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system for an explanation
        
        Args:
            question: User question
            level: Explanation complexity level
            document_id: Optional specific document to search
            max_context_length: Maximum context length
            use_cache: Whether to use cached results
            
        Returns:
            Complete explanation response
        """
        try:
            logger.info(f"Processing query: '{question}' at level '{level.value}'")
            
            # Step 1: Generate cache key
            context_hash = hashlib.md5(
                f"{question}_{document_id or 'general'}".encode()
            ).hexdigest()[:8]
            
            cache_key = self.cache_manager.explanation_key(
                question, level.value, context_hash
            )
            
            # Step 2: Check cache
            if use_cache:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    cached_result['cached'] = True
                    logger.info(f"Returning cached result for query")
                    return cached_result
            
            # Step 3: Retrieve relevant context
            retrieval_filters = None
            if document_id:
                retrieval_filters = {"document_id": document_id}
            
            relevant_docs = await self.vector_store.similarity_search(
                query=question,
                collection_name="documents",
                n_results=5,
                filter_metadata=retrieval_filters
            )
            
            if not relevant_docs:
                logger.warning(f"No relevant documents found for query: {question}")
                return {
                    'answer': "I couldn't find any relevant information to answer your question. Please try uploading a document or asking a different question.",
                    'level': level.value,
                    'source_documents': 0,
                    'context_used': "",
                    'cached': False,
                    'sources': []
                }
            
            # Step 4: Combine context with length limit
            context_parts = []
            current_length = 0
            sources_used = []
            
            for doc in relevant_docs:
                content = doc['content']
                if current_length + len(content) <= max_context_length:
                    context_parts.append(content)
                    current_length += len(content)
                    
                    # Track source information
                    source_info = {
                        'filename': doc['metadata'].get('filename', 'Unknown'),
                        'chunk_index': doc['metadata'].get('chunk_index', 0),
                        'similarity_score': round(doc.get('similarity_score', 0), 3)
                    }
                    sources_used.append(source_info)
                else:
                    # Add partial content if there's meaningful space left
                    remaining_space = max_context_length - current_length
                    if remaining_space > 200:  # Only add if meaningful
                        partial_content = content[:remaining_space].rsplit(' ', 1)[0] + "..."
                        context_parts.append(partial_content)
                        
                        source_info = {
                            'filename': doc['metadata'].get('filename', 'Unknown'),
                            'chunk_index': doc['metadata'].get('chunk_index', 0),
                            'similarity_score': round(doc.get('similarity_score', 0), 3),
                            'partial': True
                        }
                        sources_used.append(source_info)
                    break
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Step 5: Generate level-appropriate prompt
            prompt = get_prompt_for_level(level, question, context)
            
            # Step 6: Generate response from LLM
            logger.debug(f"Generating response with {len(context)} characters of context")
            response = await self.llm_provider.generate(
                prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Step 7: Prepare final result
            result = {
                'answer': response,
                'level': level.value,
                'source_documents': len(relevant_docs),
                'context_used': context if len(context) < 500 else context[:500] + "...",
                'cached': False,
                'sources': sources_used,
                'query_metadata': {
                    'question_length': len(question),
                    'context_length': len(context),
                    'response_length': len(response),
                    'processing_time': datetime.now().isoformat()
                }
            }
            
            # Step 8: Cache the result
            if use_cache:
                await self.cache_manager.set(cache_key, result)
            
            logger.info(f"Query processed successfully, response length: {len(response)}")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise Exception(f"Failed to process query: {str(e)}")
    
async def stream_query(
        self,
        question: str,
        level: ExplanationLevel,
        document_id: Optional[str] = None,
        max_context_length: int = 3000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query response for real-time display
        
        Yields:
            Chunks of the streaming response with metadata
        """
        try:
            logger.info(f"Starting streaming query: '{question}' at level '{level.value}'")
            
            # Get context (same as regular query)
            retrieval_filters = None
            if document_id:
                retrieval_filters = {"document_id": document_id}
            
            relevant_docs = await self.vector_store.similarity_search(
                query=question,
                collection_name="documents",
                n_results=5,
                filter_metadata=retrieval_filters
            )
            
            if not relevant_docs:
                yield {
                    'chunk': "I couldn't find any relevant information to answer your question.",
                    'metadata': {
                        'level': level.value,
                        'source_documents': 0,
                        'status': 'no_context'
                    }
                }
                return
            
            # Build context
            context_parts = []
            current_length = 0
            
            for doc in relevant_docs:
                content = doc['content']
                if current_length + len(content) <= max_context_length:
                    context_parts.append(content)
                    current_length += len(content)
                else:
                    remaining_space = max_context_length - current_length
                    if remaining_space > 200:
                        context_parts.append(content[:remaining_space] + "...")
                    break
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate prompt
            prompt = get_prompt_for_level(level, question, context)
            
            # Yield initial metadata
            yield {
                'chunk': '',
                'metadata': {
                    'level': level.value,
                    'source_documents': len(relevant_docs),
                    'context_length': len(context),
                    'status': 'streaming_started'
                }
            }
            
            # Stream response
            chunk_count = 0
            async for chunk in self.llm_provider.stream_generate(prompt):
                chunk_count += 1
                yield {
                    'chunk': chunk,
                    'metadata': {
                        'level': level.value,
                        'chunk_number': chunk_count,
                        'status': 'streaming'
                    }
                }
            
            # Final metadata
            yield {
                'chunk': '',
                'metadata': {
                    'level': level.value,
                    'total_chunks': chunk_count,
                    'status': 'completed'
                }
            }
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                'chunk': '',
                'metadata': {
                    'level': level.value,
                    'status': 'error',
                    'error': str(e)
                }
            }
    
async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document"""
        try:
            # Check cache first
            cached_info = await self.cache_manager.get(
                self.cache_manager.document_key(document_id, "processing")
            )
            if cached_info:
                return cached_info
            
            # Get file info
            file_info = self.file_manager.get_file_info(document_id)
            if not file_info:
                return None
            
            # Get vector store stats
            collection_stats = await self.vector_store.get_collection_stats("documents")
            
            return {
                'document_id': document_id,
                'file_info': file_info,
                'collection_stats': collection_stats,
                'status': 'processed' if file_info['exists'] else 'not_found'
            }
            
        except Exception as e:
            logger.error(f"Failed to get document info for {document_id}: {e}")
            return None
    
async def delete_document(self, document_id: str) -> bool:
        """Delete a document and clean up related data"""
        try:
            logger.info(f"Deleting document: {document_id}")
            
            # Delete file
            file_deleted = self.file_manager.delete_file(document_id)
            
            # Clear related cache entries
            await self.cache_manager.clear_pattern(f"*{document_id}*")
            
            # Note: In a production system, you'd also want to remove
            # specific document chunks from the vector store
            # This would require tracking chunk IDs by document
            
            logger.info(f"Document deletion completed: {document_id}")
            return file_deleted
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health info"""
        try:
            return {
                'vector_store': {
                    'collections': self.vector_store.list_collections(),
                    'main_collection': await self.vector_store.get_collection_stats("documents")
                },
                'storage': self.file_manager.get_storage_stats(),
                'cache': self.cache_manager.get_stats(),
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()