# RouterModelCustom

RouterModelCustom is a robust and efficient model manager and router designed to orchestrate Local Large Language Model (LLM) inference. It acts as a middleware between client applications and `llama-server` instances, providing dynamic model loading, resource management, request queuing, and an OpenAI-compatible API interface.

This project aims to simplify the deployment of multiple local LLMs by handling the complexities of process management, GPU resource allocation, and concurrent request handling.

## Table of Contents

- [RouterModelCustom](#routermodelcustom)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Configuration](#configuration)
    - [Custom Configuration](#custom-configuration)
      - [System Configuration (`system`)](#system-configuration-system)
      - [API Configuration (`api`)](#api-configuration-api)
      - [Model Configuration (`models`)](#model-configuration-models)
  - [Usage](#usage)
    - [Starting the Server](#starting-the-server)
    - [Making Requests](#making-requests)
  - [API Endpoints](#api-endpoints)
    - [OpenAI Compatible](#openai-compatible)
    - [Management \& Monitoring](#management--monitoring)
  - [Project Structure](#project-structure)

## Features

- **Dynamic Model Management**: Automatically loads and unloads models based on usage and configuration.
- **Resource Optimization**: Manages GPU/CPU resources and limits concurrent model instances.
- **Request Queuing**: Implements a queue system to handle high traffic and prevent server overload.
- **OpenAI Compatible API**: Provides a standard interface for easy integration with existing tools.
- **Health & Metrics**: Built-in endpoints for monitoring system health and performance metrics.
- **Robust Error Handling**: Includes retry mechanisms and graceful shutdown procedures.

## Installation

### Prerequisites

- Python 3.10 or higher
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (specifically the `llama-server` binary)
- NVIDIA GPU (optional, for GPU acceleration) with drivers installed.

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JohanesSetiawan/flexllama-clone.git
    cd flexllama-clone
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Prepare `llama-server`:**

    Ensure you have the `llama-server` binary compiled and available. Update the path in `config.json` (see Configuration).

## Configuration

The application is configured via `config.json`. You can start by copying the original config:

```bash
cp configOriginal.json config.json
```

### Custom Configuration

You can customize `config.json` to suit your hardware and requirements. The configuration is validated at startup to ensure stability. Below are the available options and their constraints based on the system validation logic.

#### System Configuration (`system`)

| Key | Type | Default | Constraints | Description |
|-----|------|---------|-------------|-------------|
| `llama_server_path` | string | env var | File must exist & executable | Absolute path to `llama-server` binary. |
| `max_concurrent_models` | int | 3 | 1 - 10 | Maximum number of models loaded simultaneously. |
| `idle_timeout_sec` | int | 300 | 60 - 86400 | Time in seconds before an unused model is unloaded. |
| `request_timeout_sec` | int | 300 | 30 - 3600 | Timeout for requests to `llama-server`. |
| `gpu_devices` | list[int] | [0] | - | List of GPU device IDs to use. |
| `parallel_requests` | int | 4 | 1 - 32 | Number of parallel requests per model. |
| `cpu_threads` | int | 8 | 1 - 64 | Number of CPU threads for non-GPU ops. |
| `use_mmap` | bool | true | - | Use memory mapping for model loading. |
| `flash_attention` | string | "on" | "on", "off", "auto" | Flash Attention mode. |
| `max_queue_size_per_model` | int | 100 | 10 - 1000 | Max queue size per model. |
| `queue_timeout_sec` | int | 300 | 30 - 600 | Timeout for queued requests. |

#### API Configuration (`api`)

| Key | Type | Default | Constraints | Description |
|-----|------|---------|-------------|-------------|
| `host` | string | "0.0.0.0" | - | Host to bind the server to. |
| `port` | int | 8000 | 1024 - 65535 | Port to bind the server to. |
| `cors_origins` | list[str] | ["http://localhost:3000"] | - | Allowed origins for CORS. |

#### Model Configuration (`models`)

Define your models under the `models` key. The key name serves as the model ID (alias) used in API requests.

| Key | Type | Default | Constraints | Description |
|-----|------|---------|-------------|-------------|
| `model_path` | string | Required | File must exist & end with `.gguf` | Absolute path to the `.gguf` model file. |
| `params.n_gpu_layers` | int | 99 | >= -1 | Number of layers to offload to GPU. |
| `params.n_ctx` | int | 4096 | 4096 - 8192 | Context size. |
| `params.n_batch` | int | 8 | 8 - 512 | Batch size for prompt processing. |
| `params.embedding` | bool | false | - | Set to `true` for embedding models. |
| `params.type_k` | string | "f16" | Valid quant types | Cache type for K (e.g., f16, q8_0, q4_0). |
| `params.type_v` | string | "f16" | Valid quant types | Cache type for V (e.g., f16, q8_0, q4_0). |

**Example `config.json`:**

```json
{
    "api": {
        "host": "0.0.0.0",
        "port": 8000
    },
    "system": {
        "idle_timeout_sec": 300,
        "llama_server_path": "/app/llama-server",
        "max_concurrent_models": 2,
        "gpu_devices": [0],
        "flash_attention": "on"
    },
    "models": {
        "qwen3-4b-chat": {
            "model_path": "/app/models/Qwen3-4B-Instruct.gguf",
            "params": {
                "n_gpu_layers": 99,
                "n_ctx": 4096
            }
        }
    }
}
```

## Usage

### Starting the Server

Run the application using the provided runner script:

```bash
python run.py
```

The server will start on the host and port specified in `config.json` (default: `0.0.0.0:8000`).

### Making Requests

You can interact with the server using standard HTTP clients. Here is an example using `curl` for a chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b-chat",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## API Endpoints

### OpenAI Compatible

- **`POST /v1/chat/completions`**: Chat completion endpoint. Compatible with OpenAI client libraries.
- **`POST /v1/embeddings`**: Generate embeddings for text input.
- **`GET /v1/models`**: List all available models configured in the system.

### Management & Monitoring

- **`GET /health`**: General system health check. Returns status of the router.
- **`GET /metrics`**: Prometheus-compatible metrics endpoint.
- **`GET /vram`**: Current VRAM usage status across configured GPUs.
- **`GET /v1/health/models`**: Detailed health status of all loaded models.
- **`GET /v1/queue/stats`**: Current statistics of the request queue (pending requests, processing, etc.).
- **`GET /v1/telemetry/summary`**: Summary of telemetry data (requests count, tokens generated, etc.).
- **`GET /v1/models/{model_alias}/status`**: Check the specific loading status of a model.
- **`GET /v1/models/failed`**: List models that failed to load or crashed.
- **`POST /v1/models/eject`**: Manually unload a model from memory.
    - Body: `{"model": "model_alias"}`
- **`POST /v1/models/{model_alias}/reset`**: Reset the failure state for a model, allowing it to be reloaded.

## Project Structure

```
.
├── app/
│   ├── core/           # Core logic (manager, queue, config, etc.)
│   ├── main.py         # FastAPI application entry point
│   └── ...
├── config.json         # Main configuration file
├── run.py              # Server startup script
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```