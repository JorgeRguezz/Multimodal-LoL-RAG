import json
import os
import sys
import asyncio
import numpy as np

# Adjust path to allow imports from chatbot_system if running from playground
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chatbot_system.knowledge_graph._llm import local_llm_config, shutdown_local_llm

def get_video_text_content(cache_folder_path):
    """
    Reads the kv_store_text_chunks.json file from the specified cache folder
    and returns the concatenated text content sorted by chunk order.
    """
    kv_store_path = os.path.join(cache_folder_path, "kv_store_text_chunks.json")
    
    if not os.path.exists(kv_store_path):
        print(f"Warning: {kv_store_path} does not exist.")
        return ""

    with open(kv_store_path, "r", encoding='utf-8') as f:
        kv_store = json.load(f)
    
    # The chunks are values in the dictionary.
    # We sort them by 'chunk_order_index'.
    chunks = list(kv_store.values())
    # Use get with default 0 to be safe, though schema should enforce it
    sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_order_index', 0))
    
    text_content = "\n\n".join([chunk['content'] for chunk in sorted_chunks])
    return text_content

async def summarize_text(text_content):
    """
    Summarizes the provided text content using the local LLM.
    """
    if not text_content:
        return ""
    
    # Truncate input to avoid context length issues with the small local model.
    # The local_llm_config defines 4096 max tokens. 
    # We'll use a character approximation (4 chars/token) -> ~12k chars safe limit for input.
    # Leaving room for the prompt and output.
    max_chars = 12000 
    if len(text_content) > max_chars:
        text_content = text_content[:max_chars] + "...\n(Content truncated for summarization)"

    prompt = (
        "You are a helpful assistant. Summarize the following video transcript/content into a concise paragraph "
        "that captures the main topics, entities, and narrative flow. \n\n"
        f"Content:\n{text_content}\n\n"
        "Summary:"
    )
    
    # Using cheap_model_func for summarization
    response = await local_llm_config.cheap_model_func(prompt)
    return response

async def embed_summary(summary_text):
    """
    Embeds the summary text using the local embedding model.
    """
    if not summary_text:
        return None
    
    # local_llm_config.embedding_func expects a list of strings and returns a list of numpy arrays (or single array depending on impl)
    # Based on _llm.py: local_embedding returns np.array(embeddings)
    embeddings = await local_llm_config.embedding_func([summary_text])
    
    # Return the first embedding vector
    return embeddings[0]

async def process_video_cache(cache_folder_path):
    """
    Orchestrates the extraction, summarization, and embedding for a single video cache folder.
    """
    print(f"Processing {cache_folder_path}...")
    text_content = get_video_text_content(cache_folder_path)
    if not text_content:
        print(f"No content found in {cache_folder_path}.")
        return None, None

    print("Summarizing...")
    summary = await summarize_text(text_content)
    print(f"Summary: {summary[:100]}...") # Print partial summary

    print("Embedding...")
    embedding = await embed_summary(summary)
    print("Embedding complete.")
    
    # Save the result to embedded_summary.json
    output_path = os.path.join(cache_folder_path, "embedded_summary.json")
    data = {
        "summary": summary,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    }
    
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Saved embedded summary to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    return summary, embedding

async def main():
    # Define the root cache directory
    cache_root = os.path.join(project_root, "lol_test_cache")
    
    if not os.path.exists(cache_root):
        print(f"Cache root directory not found: {cache_root}")
        return

    # Iterate over all subfolders in lol_test_cache
    subfolders = [f for f in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, f))]
    subfolders.sort()
    
    print(f"Found {len(subfolders)} folders to process.")
    
    for folder_name in subfolders:
        folder_path = os.path.join(cache_root, folder_name)
        await process_video_cache(folder_path)
        print("-" * 30)

    # Clean up LLM resources
    try:
        shutdown_local_llm()
    except Exception as e:
        print(f"Error shutting down LLM: {e}")

if __name__ == "__main__":
    asyncio.run(main())


