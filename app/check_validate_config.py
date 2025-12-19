from pathlib import Path
from app.core.config import load_config


def validate_config_file(config_path: str = "config.json"):
    """Validate config file."""
    print(f"Validating config file: {config_path}")
    print("-" * 50)

    try:
        # Load dan validate menggunakan Pydantic
        config = load_config(config_path)

        print("Config structure is valid")
        print(f"API will run on {config.api.host}:{config.api.port}")
        print(f"Llama server path: {config.system.llama_server_path}")
        print(
            f"Base models path: {config.system.base_models_path or '(not set - using absolute paths)'}")
        print(f"Idle timeout: {config.system.idle_timeout_sec} seconds")
        print(
            f"Max concurrent models: {config.system.max_concurrent_models}")
        print(f"Found {len(config.models)} model(s):")

        for alias, model_conf in config.models.items():
            print(f"  - {alias}")
            print(f"    Original Path: {model_conf.model_path}")
            resolved_path = model_conf.get_resolved_path()
            print(f"    Resolved Path: {resolved_path}")
            print(f"    Context: {model_conf.params.n_ctx}")
            print(f"    GPU Layers: {model_conf.params.n_gpu_layers}")

            # Check file size using resolved path
            model_path = Path(resolved_path)
            size_gb = model_path.stat().st_size / (1024**3)
            print(f"    Size: {size_gb:.2f} GB")

        print("-" * 50)
        print("All validations passed!")
        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        print("-" * 50)
        return False
