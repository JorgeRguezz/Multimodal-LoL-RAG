import json
import os
import sys
import asyncio
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from nano_vectordb import NanoVectorDB

# Adjust path to allow imports from chatbot_system if running from playground
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chatbot_system.knowledge_graph._llm import local_llm_config, shutdown_local_llm

# Create a lock to serialize vLLM calls because vLLM's LLM class is not thread-safe
# and _llm.py uses run_in_executor which spawns threads.
llm_lock = asyncio.Lock()

async def embed_query(query_text):
    """
    Embeds the query text using the local embedding model.
    """
    if not query_text:
        return None
    
    # local_llm_config.embedding_func returns a list of numpy arrays
    async with llm_lock:
        embeddings = await local_llm_config.embedding_func([query_text])
    return embeddings[0]

async def extract_entities(query):
    """
    Uses the local LLM to extract key named entities from the user's query.
    """

    prompt = f"""INSTRUCTION: You are a keyword extractor. Your task is to extract the most relevant keywords from the user's QUERY to help answer it.
        RULES:
        1. Keywords must be separated ONLY by a single comma (,). Do NOT use any other punctuation or line breaks.
        2. Only extract nouns, main verbs, key adjectives.
        3. Be concise and prioritize essential concepts.
        4. *IMPORTANT* Output ONLY the keywords, do NOT include any additional text such as "I've extracted the keywords for your query: (keywords)".

        --- EXAMPLES ---

        QUERY: Which animal does the protagonist encounter in the forest scene?
        OUTPUT: animal, protagonist, encounter, forest, scene

        QUERY: In the movie, what color is the car that chases the main character through the city?
        OUTPUT: movie, color, car, chases, main character, city

        QUERY: What is the weather like during the opening scene of the film?
        (A) Sunny
        (B) Rainy
        (C) Snowy
        (D) Windy
        OUTPUT: weather, opening scene, film, Sunny, Rainy, Snowy, Windy

        --- QUERY ---
        QUERY: {query}
        --- OUTPUT ---
    """

    

    async with llm_lock:
        response = await local_llm_config.best_model_func(prompt)
        print(f"---------> Entity Extraction Response: {response}")
        if ":" in response:
            split_response = response.split(":")
            clean_response = split_response[1]
        else:
            clean_response = response
    return clean_response.strip()

def load_video_embeddings(cache_root):
    """
    Iterates through the cache folders and loads the embedded_summary.json files.
    Returns a list of dicts: {'video_name': str, 'embedding': np.array, 'summary': str}
    """
    video_data = []
    
    if not os.path.exists(cache_root):
        print(f"Cache root not found: {cache_root}")
        return video_data

    subfolders = [f for f in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, f))]
    
    for folder_name in subfolders:
        summary_path = os.path.join(cache_root, folder_name, "embedded_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    video_data.append({
                        'video_name': folder_name,
                        'embedding': np.array(data['embedding']),
                        'summary': data.get('summary', '')
                    })
            except Exception as e:
                print(f"Error loading {summary_path}: {e}")
    
    return video_data

def compute_cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    # Reshape for sklearn (samples, features)
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

async def get_relevant_chunks_from_video(video_path, query_embedding, top_k_chunks=2, similarity_threshold=0.4):
    """
    Retrieves the most relevant text chunks from a specific video using vector search.
    """
    vdb_path = os.path.join(video_path, "vdb_chunks.json")
    text_store_path = os.path.join(video_path, "kv_store_text_chunks.json")

    if not os.path.exists(vdb_path) or not os.path.exists(text_store_path):
        return []

    # 1. Load Vector DB
    try:
        vdb = NanoVectorDB(embedding_dim=384, storage_file=vdb_path)
    except Exception as e:
        print(f"Error loading Chunk VDB at {vdb_path}: {e}")
        return []

    # 2. Query VDB
    results = vdb.query(query=query_embedding, top_k=top_k_chunks)
    
    if not results:
        return []

    # 3. Load Text Store
    try:
        with open(text_store_path, 'r', encoding='utf-8') as f:
            text_store = json.load(f)
    except Exception as e:
        print(f"Error loading text store at {text_store_path}: {e}")
        return []

    # 4. Retrieve and Clean Content
    chunks = []
    for res in results:
        score = res['__metrics__']
        if score < similarity_threshold:
            continue

        chunk_id = res['__id__']
        if chunk_id in text_store:
            raw_content = text_store[chunk_id]['content']
            chunks.append({
                "id": chunk_id,
                "score": score,
                "content": raw_content,
                "video_path": video_path,
                "type": "vector_search"
            })
            
    return chunks

async def get_relevant_chunks_from_graph(video_path, entity_embedding, top_k_entities=3, top_k_neighbors=5, entity_similarity_threshold=0.4):
    """
    Retrieves relevant text chunks by traversing the Knowledge Graph.
    1. Search for entities in vdb_entities.json using entity_embedding.
    2. Find 1-hop neighbors in the graph.
    3. Sort neighbors by relationship score (weight).
    4. Extract video segment IDs (chunks) from top neighbors.
    """
    vdb_entities_path = os.path.join(video_path, "vdb_entities.json")
    graph_path = os.path.join(video_path, "graph_chunk_entity_relation.graphml")
    text_store_path = os.path.join(video_path, "kv_store_text_chunks.json")

    if not os.path.exists(vdb_entities_path) or not os.path.exists(graph_path) or not os.path.exists(text_store_path):
        return []

    # 1. Search Entity VDB
    try:
        vdb = NanoVectorDB(embedding_dim=384, storage_file=vdb_entities_path)
        entity_results = vdb.query(query=entity_embedding, top_k=top_k_entities)
    except Exception as e:
        print(f"Error querying Entity VDB at {vdb_entities_path}: {e}")
        return []

    if not entity_results:
        return []

    matched_entity_names = []
    for res in entity_results:
        if res['__metrics__'] < entity_similarity_threshold:
            continue
        # The VDB stores entity_name like '"ARI"', which matches the GraphML node ID.
        matched_entity_names.append(res['entity_name'])

    # 2. Load Graph
    try:
        G = nx.read_graphml(graph_path)
    except Exception as e:
        print(f"Error loading graph at {graph_path}: {e}")
        return []

    # 3. Traverse Neighbors (1-hop) and Score
    candidate_nodes = {} # node_id -> score

    for start_node in matched_entity_names:
        # Determine the actual node ID in the graph
        # GraphML IDs often have quotes if they were saved that way.
        # We check if the exact string exists, or try to handle potential quoting issues.
        # Based on file inspection, IDs are like "\"ARI\"" in VDB and "&quot;ARI&quot;" in XML (which is "ARI" loaded).
        # Wait, "entity_name": "\"ARI\"" in JSON means the string is "ARI" (with quotes).
        # Let's try to find the node.
        
        actual_node_id = None
        if start_node in G.nodes:
            actual_node_id = start_node
        else:
            # Fallback: try stripping quotes or adding them if mismatch
            if start_node.strip('"') in G.nodes:
                actual_node_id = start_node.strip('"')
            elif f'"{start_node}"' in G.nodes:
                actual_node_id = f'"{start_node}"'
        
        if not actual_node_id:
            continue

        # Get neighbors
        neighbors = list(G.neighbors(actual_node_id))
        
        for neighbor in neighbors:
            # Get edge data
            edge_data = G.get_edge_data(actual_node_id, neighbor)
            # GraphML edge attributes are usually in a dict. 'weight' is key 'd3' or just 'weight' if parsed correctly.
            # NetworkX read_graphml usually uses the attr.name if available.
            # In the provided XML: <key id="d3" for="edge" attr.name="weight" attr.type="double" />
            # So it should be 'weight'.
            weight = edge_data.get('weight', 1.0)
            
            if neighbor not in candidate_nodes:
                candidate_nodes[neighbor] = 0.0
            candidate_nodes[neighbor] += weight

    # 4. Sort and Select Top Neighbors
    sorted_candidates = sorted(candidate_nodes.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, score in sorted_candidates[:top_k_neighbors]]

    # 5. Extract Chunk IDs
    relevant_chunk_ids = set()
    for node_id in top_nodes:
        # Get node data
        # In XML: <key id="d2" for="node" attr.name="source_id" attr.type="string" />
        node_data = G.nodes[node_id]
        source_ids = node_data.get('source_id', '')
        if source_ids:
            # Split by <SEP>
            parts = source_ids.split('<SEP>')
            for p in parts:
                if p.strip():
                    relevant_chunk_ids.add(p.strip())

    # 6. Retrieve Content
    chunks = []
    try:
        with open(text_store_path, 'r', encoding='utf-8') as f:
            text_store = json.load(f)
            
        for chunk_id in relevant_chunk_ids:
            if chunk_id in text_store:
                raw_content = text_store[chunk_id]['content']
                chunks.append({
                    "id": chunk_id,
                    "score": 1.0, # Graph retrieved chunks are "highly relevant" by relation
                    "content": raw_content,
                    "video_path": video_path,
                    "type": "knowledge_graph"
                })
    except Exception as e:
        print(f"Error loading text store for KG chunks: {e}")

    return chunks

async def generate_answer(query, context_text):
    """
    Generates an answer using the local LLM based on the provided context.
    """
    prompt = (
        "You are a helpful expert assistant. Elaborate a detailed answer to the user's question using the provided context below.\n"
        "**IMPORTANT**: The context may contain Automatic Speech Recognition (ASR) errors.\n"
        "- For example,'Ahri' might be misspelled as 'Ari', 'Harry', 'Tarry', or similar sounding words, Smolder might be misspelled as 'Smoulder', 'Smoldered', or 'Smulder'.\n"
        "- Treat these misspellings as referring to the same character.\n\n"
        "If the context doesn't contain the answer even considering these misspellings, say so.\n\n"
        "--- CONTEXT ---\n"
        f"{context_text}\n"
        "--- END CONTEXT ---\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )
    
    # Use best model for generation
    async with llm_lock:
        response = await local_llm_config.best_model_func(prompt)
    return response

async def generate_non_rag_response(query):
    """
    Generates a response using the local LLM without any context (Non-RAG).
    """
    prompt = (
        f"You are a helpful expert assistant. Answer the following question to the best of your ability.\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    async with llm_lock:
        response = await local_llm_config.best_model_func(prompt)
    return response

async def find_matching_videos_and_answer(query, cache_root, top_k=3):
    print(f"Querying for: '{query}'")
    
    # Generate Non-RAG response first
    print("\nGenerating NON-RAG response (Base Knowledge)...")
    non_rag_answer = await generate_non_rag_response(query)

    # 1. Extract Entities (Refine Query)
    print("Extracting entities from query...")
    entities_str = await extract_entities(query)
    print(f"Extracted Entities: '{entities_str}'")
    
    # 2. Embed the query AND the entities
    query_embedding = await embed_query(query)
    if query_embedding is None:
        print("Failed to embed query.")
        return

    entity_embedding = await embed_query(entities_str)
    if entity_embedding is None:
        print("Failed to embed entities. Using query embedding for graph search.")
        entity_embedding = query_embedding

    # 3. Load video embeddings and find matching videos
    videos = load_video_embeddings(cache_root)
    print(f"Loaded embeddings for {len(videos)} videos.")

    if not videos:
        print("No video embeddings found.")
        return

    video_results = []
    for vid in videos:
        score = compute_cosine_similarity(query_embedding, vid['embedding'])
        if score > 0.4:            # filter by threshold
            video_results.append({
                'video_name': vid['video_name'],
                'score': score,
                'summary': vid['summary']
            })

    video_results.sort(key=lambda x: x['score'], reverse=True)
    top_videos = video_results[:top_k]

    print("\n--- Matching Videos ---")
    for i, res in enumerate(top_videos):
        print(f"{i+1}. {res['video_name']} (Score: {res['score']:.4f})")

    # 4. Retrieve chunks (Vector Search + Knowledge Graph)
    print("\nRetrieving relevant chunks (Vector + Graph)...")
    all_relevant_chunks = []
    seen_chunk_ids = set()

    for vid in top_videos:
        video_path = os.path.join(cache_root, vid['video_name'])
        
        # A. Vector Search on Text Chunks (Standard RAG)
        vector_chunks = await get_relevant_chunks_from_video(video_path, query_embedding, top_k_chunks=25)
        for c in vector_chunks:
            if c['id'] not in seen_chunk_ids:
                c['source_video'] = vid['video_name']
                all_relevant_chunks.append(c)
                seen_chunk_ids.add(c['id'])
        
        # B. Knowledge Graph Search (Graph RAG)
        graph_chunks = await get_relevant_chunks_from_graph(video_path, entity_embedding, top_k_entities=25, top_k_neighbors=10)
        for c in graph_chunks:
            if c['id'] not in seen_chunk_ids:
                c['source_video'] = vid['video_name']
                all_relevant_chunks.append(c)
                seen_chunk_ids.add(c['id'])

    if not all_relevant_chunks:
        print("No detailed chunks found.")
        return

    # Debug: Print retrieved chunks
    print(f"\n--- Debug: Retrieved {len(all_relevant_chunks)} Chunks ---")
    for i, chunk in enumerate(all_relevant_chunks):
        # Truncate content for display
        display_content = chunk['content'][:150].replace(chr(10), ' ') + "..."
        print(f"[{i+1}] Source: {chunk['source_video']} | Type: {chunk['type']} | Score: {chunk['score']:.4f}")
        print(f"Content: {display_content}")
        print("-" * 10)
    print("--- End Debug ---\n")

    # 5. Format Context
    print(f"Collected {len(all_relevant_chunks)} chunks. Generating answer...")
    
    context_parts = []

    for i, chunk in enumerate(all_relevant_chunks):
        # Truncate content to prevent context overflow (approx 500 tokens per chunk)
        content = chunk['content']
        if len(content) > 2000:
            content = content[:2000] + "... (truncated)"
        
        if i > 0 and (f"Source Video: {chunk['source_video']}" in context_parts[i-1] or f"(Continued from previous source video)" in context_parts[i-1]):
            context_parts.append(f"(Continued from previous source video)\n{content}\n")
        else:
            context_parts.append(
                f"Source Video: {chunk['source_video']}\n"
                f"Content:\n{content}\n"
            )
    
    full_context = "\n\n".join(context_parts)

    print(f"\n--- Full Context for Answer Generation ---\n{full_context}\n--- End Full Context ---\n")

    # 6. Generate Answer
    try:
        answer = await generate_answer(query, full_context)
        print("\n" + "="*30)
        print("FINAL ANSWER (RAG):")
        print("="*30)
        print(answer)
        print("="*30)
    except Exception as e:
        print(f"Error generating answer: {e}")

    # Print Non-RAG Response at the end
    print("\n" + "="*30)
    print("NON-RAG RESPONSE (Base Knowledge):")
    print("="*30)
    print(non_rag_answer)
    print("="*30)

async def main():
    cache_root = os.path.join(project_root, "lol_test_cache")
    
    # Get query from command line args or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How do I play Ahri?" # Default test query
        print(f"No query provided. Using default: '{query}'")

    try:
        await find_matching_videos_and_answer(query, cache_root)
    finally:
        # Clean up resources
        try:
            shutdown_local_llm()
        except Exception as e:
            print(f"Error shutting down LLM: {e}")

if __name__ == "__main__":
    asyncio.run(main())