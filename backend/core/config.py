# Settings and config
# This is the central configuration file that manages all app settings. 
# It uses Pydantic for validation and supports environment variables from .env files.
from pydantic import BaseSettings
from typing import Optional,List
import os

class Settings(BaseSettings):
    # General settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ELI5"
    VERSION: str = "0.1.0"

    # LLM API settings
    GROQ_API_KEY: Optional[str] = None
    QROQ_MODEL: str = "mixtral-8x7b-32768"
    USE_LOCAL_MODEL: bool = False
    LOCAL_MODEL_PATH: Optional[str] = None
    
    # vector database settings
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Rate limiting settings
    RATE_LIMIT_PER_MINUTE: int = 10
    MAX_CONCURRENT_REQUESTS: int = 5
    
    # file upload settings
    MAX_FILE_SIZE_MB: int = 10 * 1024 * 1024  # 10 MB
    UPLOAD_FOLDER: str = "./uploads"
    ALLOWED_FILE_TYPES: List[str] = ["txt", "pdf", "docx"]
    
    # cache settings
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour

    # Security (later)
    SECRET_KEY: str = "change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Global settings instance
settings = Settings()
