# Project Overview

This project is a sophisticated, multi-component chatbot system built with Python. It's designed to be modular and extensible, leveraging local Large Language Models (LLMs) and Vision-Language Models (VLMs) to provide intelligent and context-aware responses.

The core architecture consists of:
- A **Main Chatbot Application** (`mcp_chatbot.py`): This is the central orchestrator. It handles user interaction, manages conversation history, and connects to various tools and services using the Multi-Context Protocol (MCP).
- A **Vision-Language Model (VLM) API** (`smolvlm2_api.py`): A standalone Flask server that provides access to the `SmolVLM2` model for image and video analysis. The main chatbot communicates with this API via HTTP requests.
- **External Tools** (e.g., `videogame_search_tool.py`): These are independent Python scripts that run as MCP servers, offering specialized functionalities that the main chatbot can call upon.
- **GPU Management** (`gpu_manager.py`): A utility to efficiently load and manage the AI models on the GPU, handling potential memory issues.

The system is designed to run in separate, isolated environments to handle conflicting dependencies between the LLM (`vllm`) and VLM (`transformers`) libraries.

# Building and Running

The system requires at least two separate Python virtual environments due to library incompatibilities.

### Environment 1: Main Chatbot (LLM)

This environment runs the main chatbot application.

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv_llm
    source venv_llm/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r chatbot_system/requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root and add any required API keys. For the video game search tool, you'll need a RAWG.io API key.
    ```
    RAWG_API_KEY="your_rawg_api_key_here"
    ```

4.  **Run the main chatbot:**
    This script will also launch the MCP tools defined in `server_config.json`.
    ```bash
    cd chatbot_system
    python mcp_chatbot.py
    ```

### Environment 2: SmolVLM2 API

This environment runs the Flask API for the vision-language model.

1.  **Create and activate a separate virtual environment:**
    ```bash
    python3 -m venv venv_smolvlm
    source venv_smolvlm/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r chatbot_system/smolvlm2_api_requirements.txt
    ```

3.  **Run the API server:**
    ```bash
    cd chatbot_system
    python smolvlm2_api.py
    ```
    The API will be available at `http://localhost:5000`.

# Development Conventions

*   **Modular Architecture:** The project is split into distinct components that communicate over well-defined protocols (MCP and HTTP). This is the primary architectural pattern.
*   **Dependency Management:** Separate `requirements.txt` files are used for components with conflicting dependencies. Virtual environments are essential for development and execution.
*   **Configuration:** Server configurations are externalized into `server_config.json`. API keys and other secrets are managed via a `.env` file.
*   **Tool Integration:** New functionalities are added by creating standalone MCP server tools. The main chatbot discovers and integrates with them through the `server_config.json` file.
*   **Code Style:** The code is generally well-structured with classes and functions. It includes inline comments and diagnostic print statements for debugging.
*   **Experimentation:** The `playground/` directory is used for testing, debugging, and experimenting with new features before integrating them into the main application.
