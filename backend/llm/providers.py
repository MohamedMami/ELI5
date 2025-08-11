# Groq, local models 
# llm integration
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncGenerator
from groq import AsyncGroq
import asyncio
import logging
from core.config import settings

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    # base for all LLM providers
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        pass
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        pass
class GroqProvider(LLMProvider):
    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set in settings")
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = await self.client.chat.completions.create(
                model= kwargs.get('model', settings.QROQ_MODEL),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq api error: {e}")
            raise Exception(f"Failed to generate response: {e}")
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        
        try:
            stream = await self.client.chat.completions.create(
                model=kwargs.get('model', settings.QROQ_MODEL),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            raise Exception(f"Failed to stream response: {e}")
        
class LocalModelProvider(LLMProvider):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: Implement local model loading using transformers or llama.cpp
        logger.warning(f"Using local model at {self.model_path}. Ensure it is properly set up.")
    async def generate(self, prompt: str, **kwargs) -> str:
        return "Generated text from local model"    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "Streaming text from local model"
        
def get_llm_provider() -> LLMProvider:
    if settings.USE_LOCAL_MODEL and settings.LOCAL_MODEL_PATH:
        return LocalModelProvider(settings.LOCAL_MODEL_PATH)
    elif settings.GROQ_API_KEY:
        return GroqProvider()
    else:
        raise ValueError("No valid LLM provider configured. Set either USE_LOCAL_MODEL or GROQ_API_KEY in settings.")
    
llm_provider = get_llm_provider()