import os
import json
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)  # Validasi port range
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Daftar origin yang diizinkan untuk CORS"
    )


class SystemConfig(BaseModel):
    enable_idle_timeout: bool = Field(
        default=True,
        description="Enable/disable idle timeout. Set False untuk GPU kuat (model tetap loaded), True untuk GPU terbatas (eject model saat idle)."
    )

    idle_timeout_sec: int = Field(
        default=300, ge=60, le=86400,  # Minimal 60 detik, maksimal 24 jam
        description="Waktu idle sebelum 'Cold Sleep'. Hanya berlaku jika enable_idle_timeout=True."
    )

    llama_server_path: str = Field(
        default=os.getenv("LLAMA_SERVER_PATH", ""),
        description="Path absolut ke binary llama-server."
    )

    max_concurrent_models: int = Field(
        default=3, ge=1, le=10,
        description="Maksimum model yang bisa running bersamaan."
    )

    request_timeout_sec: int = Field(
        default=300, ge=30, le=3600,
        description="Timeout untuk request ke llama-server (detik)"
    )

    preload_models: list[str] = Field(
        default=[],
        description="Model aliases untuk di-preload saat startup. "
                    "Gunakan ['*'] untuk load semua model, atau "
                    "['model1', 'model2'] untuk model spesifik."
    )

    preload_delay_sec: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Delay (detik) antar preload model untuk menghindari VRAM overflow. "
                    "Berguna saat preload multiple models."
    )

    min_vram_required: int = Field(
        default=500,
        ge=200,
        le=750,
        description="Digunakan untuk membatasi ukuran VRAM pada GPU (dalam MB)"
    )

    vram_multiplier: float = Field(
        default=1.1,
        ge=1.0,
        le=3.0,
        description="Multiplier untuk estimasi VRAM yang dibutuhkan saat load model"
    )

    keep_warm_models: int = Field(
        default=2,
        ge=0,
        description="Jumlah model paling populer yang tetap warm"
    )

    gpu_devices: list[int] = Field(
        default=[0],
        description="List of GPU device indices to use"
    )

    parallel_requests: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Jumlah parallel requests per model (llama.cpp --parallel)"
    )

    cpu_threads: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Jumlah CPU threads untuk non-GPU ops (llama.cpp --threads)"
    )

    use_mmap: bool = Field(
        default=True,
        description="Gunakan memory mapping untuk loading model (faster tapi butuh stable VRAM)"
    )

    flash_attention: str = Field(
        default="on",
        description="Flash Attention mode: 'on', 'off', atau 'auto'"
    )

    @property
    def calculated_max_queue_size(self) -> int:
        """Calculate optimal queue size based on timeout and expected latency."""
        # Assume worst case: 10s per request (conservative)
        return max(20, int(self.queue_timeout_sec / 10) * 2)

    max_queue_size_per_model: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maksimum queue size per model"
    )

    queue_timeout_sec: int = Field(
        default=300,
        ge=30,
        le=600,
        description="Timeout untuk request di queue"
    )

    timeout_warmup_sec: int = Field(
        default=180,
        ge=120,
        le=3600,
        description="Digunakan untuk runner pada warmup di load_single_model"
    )

    wait_ready_sec: int = Field(
        default=120,
        ge=120,
        le=3600,
        description="Digunakan untuk menunggu status ready setelah mendapatkan runner"
    )

    @field_validator('llama_server_path')
    @classmethod
    def validate_llama_server_path(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "llama_server_path harus diisi, atau set environment variable LLAMA_SERVER_PATH")
        path = Path(v)
        if not path.exists():
            raise ValueError(f"llama-server tidak ditemukan di path: {v}")
        if not path.is_file():
            raise ValueError(f"llama_server_path bukan file: {v}")
        if not os.access(path, os.X_OK):
            raise ValueError(f"llama-server tidak executable: {v}")
        return v


class ModelParams(BaseModel):
    n_gpu_layers: int = Field(default=99, ge=-1)
    n_ctx: int = Field(default=4096, ge=512, le=131072)
    n_batch: int = Field(default=8, ge=8, le=512)
    rope_freq_base: Optional[int] = Field(
        default=None,
        ge=0,
        description="RoPE frequency base. None = gunakan default model"
    )
    embedding: bool = False
    chat_template: Optional[str] = None

    parallel_override: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Override parallel requests untuk model ini"
    )

    batch_override: Optional[int] = Field(
        default=None,
        ge=128,
        le=4096,
        description="Override batch size untuk model ini"
    )

    type_k: Optional[str] = Field(
        default="f16",
        description="Cache type untuk K (contoh: f16, q8_0, q4_0)"
    )

    type_v: Optional[str] = Field(
        default="f16",
        description="Cache type untuk V (contoh: f16, q8_0, q4_0)"
    )

    @field_validator('type_k', 'type_v')
    @classmethod
    def validate_cache_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return "f16"  # Default to f16 if not specified
        valid_types = ['f16', 'f32', 'bf16', 'q8_0',
                       'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']
        if v not in valid_types:
            raise ValueError(
                f"Cache type harus salah satu dari: {', '.join(valid_types)}"
            )
        return v


class ModelConfig(BaseModel):
    model_path: str = Field(..., description="Path absolut ke file .gguf.")
    params: ModelParams = Field(default_factory=ModelParams)

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model file tidak ditemukan: {v}")
        if not path.is_file():
            raise ValueError(f"Model path bukan file: {v}")
        if not v.endswith('.gguf'):
            raise ValueError(f"Model file harus berformat .gguf: {v}")
        return v


class AppConfig(BaseModel):
    api: ApiConfig
    system: SystemConfig
    models: Dict[str, ModelConfig]

    @field_validator('models')
    @classmethod
    def validate_models_not_empty(cls, v: Dict[str, ModelConfig]) -> Dict[str, ModelConfig]:
        if not v:
            raise ValueError("Minimal harus ada satu model terdefinisi")
        return v


def load_config(path: str) -> AppConfig:
    """Membaca file config.json dan memvalidasinya menggunakan Pydantic."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        config = AppConfig(**data)
        print(f"Configurasi '{path}' berhasil di-load dan divalidasi.")
        print(f"Model tersedia: {list(config.models.keys())}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File config.json tidak ditemukan di '{path}'")
    except json.JSONDecodeError as e:
        raise ValueError(f"config.json bukan JSON valid: {e}")
    except ValueError as e:
        raise ValueError(f"Validasi config gagal: {e}")
    except Exception as e:
        raise RuntimeError(f"Error saat membaca config.json: {e}")
