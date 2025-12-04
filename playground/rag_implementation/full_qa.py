import os
import sys
import asyncio
import json
import numpy as np
from dataclasses import dataclass

# Adjust path to allow imports from chatbot_system if running from playground
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chatbot_system.knowledge_graph._llm import local_llm_config, shutdown_local_llm
from chatbot_system.knowledge_graph._storage.kv_json import JsonKVStorage
from chatbot_system.knowledge_graph._storage.vdb_nanovectordb import NanoVectorDBStorage, NanoVectorDBVideoSegmentStorage
from chatbot_system.knowledge_graph._storage.gdb_networkx import NetworkXStorage
from chatbot_system.knowledge_graph.base import QueryParam
import chatbot_system.knowledge_graph._op as op_module
from playground.rag_implementation.query_video_match import find_matching_videos, load_video_embeddings, compute_cosine_similarity, embed_query

# --- Monkey Patching ---
# We patch retrieved_segment_caption to avoid running VLM and needing video files.
# Instead, we just return the existing content from the video_segments DB.
def mock_retrieved_segment_caption(caption_model, caption_tokenizer, keywords, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    results = {}
    for s_id in retrieved_segments:
        # Parse s_id to get video_name and index
        # Format is usually "{video_name}_{index}"
        # But video_name might contain underscores.
        # The standard splitting in the codebase is:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        
        # Access the data
        if video_name in video_segments._data and index in video_segments._data[video_name]:
            results[s_id] = video_segments._data[video_name][index]["content"]
        else:
            results[s_id] = "Content not found."
    return results

op_module.retrieved_segment_caption = mock_retrieved_segment_caption

# --- Main QA Logic ---

async def main():
    cache_root = os.path.join(project_root, "lol_test_cache")
    
    # 1. Get User Query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Who is Ahri and what is her story?"
        print(f"No query provided. Using default: '{query}'")

    # 2. Find Best Matching Video
    # Re-implementing simple match logic here to return the actual object
    query_embedding = await embed_query(query)
    if query_embedding is None:
        print("Error embedding query.")
        return
    
    videos = load_video_embeddings(cache_root)
    if not videos:
        print("No video embeddings found.")
        return

    results = []
    for vid in videos:
        score = compute_cosine_similarity(query_embedding, vid['embedding'])
        results.append({
            'video_name': vid['video_name'],
            'score': score
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    best_video = results[0]
    print(f"\nSelected Video: {best_video['video_name']} (Score: {best_video['score']:.4f})")
    
    # 3. Setup Environment for the Selected Video
    video_cache_dir = os.path.join(cache_root, best_video['video_name'])
    
    # Construct global_config expected by storage classes
    global_config = {
        "working_dir": video_cache_dir,
        "llm": {
            "embedding_batch_num": 32,
            "embedding_func_max_async": 4,
            "best_model_func": local_llm_config.best_model_func,
            "cheap_model_func": local_llm_config.cheap_model_func,
            "cheap_model_max_token_size": 4096,
        },
        "query_better_than_threshold": 0.2,
        "video_embedding_batch_num": 32,
        "video_embedding_dim": 1024, # ImageBind usually 1024
        "segment_retrieval_top_k": 2,
        "retrieval_topk_chunks": 3,
        "fine_num_frames_per_segment": 5, # Unused by mock but required by dict lookups
        "llm_response_cache": None # Disable cache for now
    }
    
    # Inject embedding func into llm config for storage classes that use it
    # The storage classes expect `self.embedding_func` to be passed or derived.
    # NanoVectorDBStorage uses `self.embedding_func.embedding_dim`.
    # We need to patch/pass this.
    
    # In extractor.py, it passes `embedding_func` to constructor.
    # But `videorag_query` expects *instances*.
    
    print("Initializing databases...")
    
    # Helper to mock embedding func object with attributes
    class MockEmbeddingFunc:
        def __init__(self, func, dim):
            self.func = func
            self.embedding_dim = dim
            
        async def __call__(self, texts):
            return await self.func(texts)

    wrapped_embedding_func = MockEmbeddingFunc(local_llm_config.embedding_func, 384)

    # Initialize Storage Instances
    entities_vdb = NanoVectorDBStorage(
        namespace="entities",
        global_config=global_config,
        embedding_func=wrapped_embedding_func,
        meta_fields={"entity_name"}
    )
    
    text_chunks_db = JsonKVStorage(
        namespace="text_chunks",
        global_config=global_config
    )
    
    chunks_vdb = NanoVectorDBStorage(
        namespace="chunks",
        global_config=global_config,
        embedding_func=wrapped_embedding_func
    )
    
    video_path_db = JsonKVStorage(
        namespace="video_path",
        global_config=global_config
    )
    
    video_segments = JsonKVStorage(
        namespace="video_segments",
        global_config=global_config
    )
    
    # Mock Video Segment Storage to avoid loading ImageBind (CUDA OOM)
    class MockVideoSegmentStorage(NanoVectorDBVideoSegmentStorage):
        def __init__(self, namespace, global_config, embedding_func):
            self.namespace = namespace
            self.global_config = global_config
            self.embedding_func = embedding_func
            
        async def query(self, query):
            print("Skipping Visual Retrieval (ImageBind) to save memory.")
            return []

    video_segment_feature_vdb = MockVideoSegmentStorage(
        namespace="video_segments", 
        global_config=global_config,
        embedding_func=wrapped_embedding_func
    )
    
    knowledge_graph_inst = NetworkXStorage(
        namespace="chunk_entity_relation",
        global_config=global_config
    )
    
    # 4. Execute VideoRAG Query
    print("Executing VideoRAG Query...")
    
    query_param = QueryParam(
        top_k=3,
        response_type="Detailed Answer"
    )
    # Manually set attributes that might be missing from __init__ or are class vars
    query_param.naive_max_token_for_text_unit = 2000
    query_param.wo_reference = False
    
    try:
        response = await op_module.videorag_query(
            query=query,
            entities_vdb=entities_vdb,
            text_chunks_db=text_chunks_db,
            chunks_vdb=chunks_vdb,
            video_path_db=video_path_db,
            video_segments=video_segments,
            video_segment_feature_vdb=video_segment_feature_vdb,
            knowledge_graph_inst=knowledge_graph_inst,
            caption_model=None, # Mocked
            caption_tokenizer=None, # Mocked
            query_param=query_param,
            global_config=global_config
        )
        
        print("\n" + "="*30)
        print("FINAL ANSWER:")
        print("="*30)
        print(response)
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"Error during VideoRAG query: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            shutdown_local_llm()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
