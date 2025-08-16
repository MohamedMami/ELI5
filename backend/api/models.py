# Pydantic models
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from prompts.level_prompts import ExplanationLevel

# Request Models
class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    original_filename: str
    file_size: int
    file_type: str
    chunks_created: int
    processing_status: str
    message: str

class QuestionRequest(BaseModel):
    """Request model for explanation queries"""
    question: str = Field(
        ..., 
        min_length=5, 
        max_length=500,
        description="The question to be explained"
    )
    level: ExplanationLevel = Field(
        ...,
        description="Explanation complexity level"
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional specific document to search in"
    )
    use_cache: bool = Field(
        True,
        description="Whether to use cached results"
    )
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

# Response Models
class SourceInfo(BaseModel):
    """Information about a source document"""
    filename: str
    chunk_index: int
    similarity_score: float
    partial: bool = False

class QueryMetadata(BaseModel):
    """Metadata about query processing"""
    question_length: int
    context_length: int
    response_length: int
    processing_time: str

class ExplanationResponse(BaseModel):
    """Response model for explanations"""
    answer: str = Field(..., description="The generated explanation")
    level: str = Field(..., description="Explanation level used")
    source_documents: int = Field(..., description="Number of source documents used")
    context_used: str = Field(..., description="Context snippet used (truncated)")
    cached: bool = Field(..., description="Whether result was cached")
    sources: List[SourceInfo] = Field(..., description="Source document information")
    query_metadata: QueryMetadata = Field(..., description="Query processing metadata")

class StreamChunk(BaseModel):
    """Model for streaming response chunks"""
    chunk: str = Field(..., description="Text chunk")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")

class DocumentInfo(BaseModel):
    """Information about a processed document"""
    document_id: str
    original_filename: str
    file_size: int
    upload_time: str
    file_type: str
    chunks_created: Optional[int] = None
    processing_status: str
    exists: bool

class SystemStats(BaseModel):
    """System statistics and health"""
    vector_store: Dict[str, Any]
    storage: Dict[str, Any]
    cache: Dict[str, Any]
    status: str
    timestamp: str

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str
    version: str
    services: Dict[str, str]

class AvailableLevels(BaseModel):
    """Available explanation levels"""
    levels: List[Dict[str, str]]

# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "validation_error"
    detail: str
    field_errors: Optional[List[Dict[str, Any]]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Rate Limit Models
class RateLimitResponse(BaseModel):
    """Rate limit exceeded response"""
    error: str = "rate_limit_exceeded"
    detail: str
    retry_after: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())