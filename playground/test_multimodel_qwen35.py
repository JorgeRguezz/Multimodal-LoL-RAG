import torch
import os 
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time
import json

# 1. Download the quantized GGUF model
model_repo = "unsloth/Qwen3.5-27B-GGUF"
model_filename = "Qwen3.5-27B-GGUF-Q8_0.gguf"

local_model_dir = "/media/gatv-projects/ssd/AI_models" 

print(f"Downloading or loading {model_filename} from cache...")
model_path = hf_hub_download(
    repo_id = model_repo,
    filename = model_filename, 
    local_dir = local_model_dir,
    local_dir_use_symlinks = False  # Avoid symlinks for better compatibility
)

print(f"Model is stored at: {model_path}")

# 2. Load the model into the GPU
print("Loading model into GPU...")
llm = Llama(
    model_path = model_path,
    n_ctx = 16384,
    n_gpu_layers = None, # TODO
    verbose = False, # llama-cpp-python will print only loading info; set to True for more detailed logs
    n_batch=2048
)

knowledge_path = "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/extracted_data/S+_BUFFED_AHRI_MID_IS_GOD_TIER_Best_Build_&_Runes_How_to_Carry_with_Ahri_League_of_Legends/kv_store_video_frames.json"

with open(knowledge_path, "r") as f:
    knowledge_json = json.load(f)

# 1. Flatten all segments into a single, easy-to-manage list
all_segments = []
for video in knowledge_json.values():
    for segment in video.values():
        all_segments.append(segment["vlm_output"])

# 2. Define static variables OUTSIDE the loop for better performance
system_prompt = """
    You are an expert on summarizing League of Legends gameplay based on visual descriptions.
    Given the VLM outputs describing the visual content of consecutive video segments, your task is to generate a concise summary of the key events and actions.
    Focus on identifying the main champion(s) involved, their actions, and any significant interactions.
    Use the VLM output as your primary source of information, and infer the most likely gameplay events based on the visual cues provided. Your summary should be clear, informative, and capture the essence of the gameplay.
"""

chunk_size = 5
batch_number = 0
total_time = 0

# 3. Iterate through the list in chunks of 4 using Python slicing
for i in range(0, len(all_segments), chunk_size):
    # Grab 4 segments at a time (or however many are left at the end of the list)
    chunk = all_segments[i:i + chunk_size]
    
    # 4. Merge the 4 text descriptions into one big text block
    merged_content = "\n\n--- Next Segment ---\n\n".join(chunk)
    
    print(f"-----> Inferencing batch {batch_number} (contains {len(chunk)} segments) with LLM...")
    
    prompt = f""" 
    Based on the following VLM descriptions of a League of Legends gameplay sequence, describe what is happening overall.
    
    VLM Outputs of the merged segments:
    {merged_content}
    """

    # 3. Setup the Knowledge Graph Extraction test prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ] 

    print("Generating response...")

    # 4. Generate text (using recommended params)
    start_inference = time.time()
    reponse = llm.create_chat_completion(
        messages = messages,
        temperature=0.7,
        top_p = 1.0,
        min_p = 0.01,
        repeat_penalty = 1.0,
        max_tokens = 8000
    )
    end_inference = time.time()

    batch_number += 1

    print(f"Inference time: {end_inference - start_inference:.2f} seconds")
    print("\n=== Generated Response ===")
    print(reponse["choices"][0]["message"]["content"])