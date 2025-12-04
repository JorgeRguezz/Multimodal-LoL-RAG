import numpy as np
import os
import asyncio
import google.generativeai as genai
from dataclasses import asdict, dataclass, field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from dotenv import load_dotenv

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc
from ._llm import local_embedding, LLMConfig # Re-using the same config class structure

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini API calls will fail.")

# A basic retry mechanism for API calls
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception), # A generic retry for any API error
)
async def gemini_complete_if_cache(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    
    # Build Gemini-compatible message history
    messages = []
    # Gemini doesn't have a system role, prepend to first user message
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    for msg in history_messages:
        # Translate roles: 'assistant' -> 'model'
        role = 'model' if msg['role'] == 'assistant' else 'user'
        messages.append({'role': role, 'parts': [msg['content']]})
    
    messages.append({'role': 'user', 'parts': [prompt]})


    if hashing_kv is not None:
        args_hash = compute_args_hash(model_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    model = genai.GenerativeModel(model_name)
    response = await model.generate_content_async(messages)
    
    response_text = response.text

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model_name}}
        )
        await hashing_kv.index_done_callback()
    
    return response_text


async def gemini_1_5_flash_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

gemini_config = LLMConfig(
    embedding_func_raw=local_embedding,
    embedding_model_name="all-MiniLM-L6-v2",
    embedding_dim=384,
    embedding_max_token_size=512,
    embedding_batch_num=32,
    embedding_func_max_async=4,
    query_better_than_threshold=0.2,

    # LLM for generation
    best_model_func_raw=gemini_1_5_flash_complete,
    best_model_name="gemini-2.5-flash",    
    best_model_max_token_size=8192, # Check model context window
    best_model_max_async=16,
        
    # Using the same model for both "best" and "cheap" for simplicity
    cheap_model_func_raw=gemini_1_5_flash_complete,
    cheap_model_name="gemini-2.5-flash",
    cheap_model_max_token_size=8192,
    cheap_model_max_async=16
)
