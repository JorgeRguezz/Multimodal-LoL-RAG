# Chatbot System Documentation

This document provides an overview of the different components of the chatbot system.

## Set-up

Currently running the system in two different environments. One is for the LLM and its dependecies, the other one is for the VLM. In the future the goal is to find an environment that works for both. Right now, the set up should be one environment to run the main script with the LLM and another environment to run the smolvlm2_api.py. The reason of this is an incompatibility between the transformers library used by the VLM and the vLLM library used by the LLM.

## Components

### `chatbot_system/gpu_manager.py`

This module is responsible for managing the allocation of models on the GPU. The `GPUModelManager` class handles the loading of both the Large Language Model (LLM) and the Vision-Language Model (VLM) into the GPU's memory. It is designed to load models sequentially and includes cleanup mechanisms to free up GPU memory if an out-of-memory error occurs or when the application shuts down.

### `chatbot_system/mcp_chatbot.py`

This is the core of the chatbot application. It orchestrates the entire system, from model loading to user interaction. Key functionalities include:
- **Model Initialization**: Uses `GPUModelManager` to load the LLM and VLM.
- **MCP Server Connection**: Connects to external tools (like the video game search) defined in `server_config.json` using the Multi-Context Protocol (MCP).
- **Query Processing**: Handles user input, which can be a text query or a URL to a YouTube video.
- **Tool Integration**: Detects and calls external tools (e.g., `@search_video_games(...)`) when the user's query or the LLM's response requires it.
- **VLM Integration**: If a YouTube URL is provided, it uses `VLMProcessor` to download and analyze the video content.
- **Interactive Loop**: Provides a command-line interface for continuous user interaction.

### `chatbot_system/server_config.json`

This JSON file configures the MCP servers that the chatbot can connect to. It maps a server name to the command required to start it. For example, it tells the main chatbot application how to launch the `videogame_search_tool.py` script.

### `chatbot_system/smolvlm2_api.py`

This script creates a standalone Flask-based web API for the `SmolVLM2` model. It exposes a `/generate` endpoint that accepts a conversation history (including text and images) and returns a generated text response from the model. It also provides a `/health` endpoint to check the status of the service. This allows other applications to interact with the VLM over HTTP.

### `chatbot_system/videogame_search_tool.py`

This is an external tool that runs as an MCP server. It provides a `search_video_games` function that the main chatbot can call. The tool takes a game title as input, queries the RAWG.io API for game details, and returns the information in a structured JSON format. This allows the chatbot to answer questions about video games.

### `chatbot_system/vlm_processor.py`

This module encapsulates the logic for the Vision-Language Model (VLM). The `VLMProcessor` class is responsible for:
- Loading the `SmolVLM2` model and processor.
- Downloading videos from YouTube using `yt-dlp`.
- Analyzing a local video file based on a user's text query and generating a relevant textual description of the video's content.
