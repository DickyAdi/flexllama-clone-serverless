import os
import json
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ApiConfig(BaseModel):
    """Konfigurasi API server untuk router model."""

    host: str = Field(
        default="0.0.0.0",
        description="IP address untuk binding API server. "
                    "Gunakan '0.0.0.0' untuk menerima koneksi dari semua interface, "
                    "atau '127.0.0.1' untuk localhost only."
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port untuk API server. Range valid: 1024-65535. "
                    "Pastikan port tidak digunakan oleh service lain."
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Daftar origin yang diizinkan untuk CORS (Cross-Origin Resource Sharing). "
                    "Tambahkan URL frontend/backend yang akan mengakses API ini. "
                    "Contoh: ['http://localhost:3000', 'https://myapp.com']"
    )


class SystemConfig(BaseModel):
    """
    Konfigurasi sistem untuk router model.

    Mengatur behavior llama-server, VRAM management, queue system,
    dan parameter performa lainnya.
    """

    enable_idle_timeout: bool = Field(
        default=True,
        description="Enable/disable auto-unload model saat idle. "
                    "• True: Model akan di-unload dari VRAM setelah idle_timeout_sec (hemat VRAM, cocok untuk GPU terbatas). "
                    "• False: Model tetap loaded di VRAM selamanya (response cepat, cocok untuk GPU besar atau production)."
    )

    idle_timeout_sec: int = Field(
        default=300,
        ge=60,
        le=86400,
        description="Waktu idle (detik) sebelum model di-unload dari VRAM. "
                    "Hanya berlaku jika enable_idle_timeout=True. "
                    "• 60-300: Untuk development/testing. "
                    "• 300-900: Untuk production dengan traffic sedang. "
                    "• 900+: Untuk traffic rendah, hemat resource."
    )

    llama_server_path: str = Field(
        default=os.getenv("LLAMA_SERVER_PATH", ""),
        description="Path absolut ke binary llama-server dari llama.cpp. "
                    "Bisa di-override dengan environment variable LLAMA_SERVER_PATH. "
                    "Contoh: /home/user/llama.cpp/build/bin/llama-server"
    )

    base_models_path: str = Field(
        default=os.getenv("BASE_MODELS_PATH", ""),
        description="Base directory untuk model files (.gguf). "
                    "Jika di-set, model_path di setiap model bisa menggunakan relative path. "
                    "Bisa di-override dengan environment variable BASE_MODELS_PATH. "
                    "Jika kosong, semua model_path harus menggunakan absolute path."
    )

    max_concurrent_models: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maksimum jumlah model yang bisa running bersamaan di VRAM. "
                    "Sesuaikan dengan kapasitas VRAM GPU Anda. "
                    "• 1-2: GPU 8-12GB. "
                    "• 3-5: GPU 24GB. "
                    "• 5+: Multi-GPU atau GPU 48GB+."
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
        default=5,  # Reduced from 30s - faster startup for small models
        ge=1,
        le=120,  # Reduced max from 300 to 120
        description="Delay (detik) antar preload model untuk menghindari VRAM overflow. "
                    "Berguna saat preload multiple models."
    )

    min_vram_required: int = Field(
        default=500,
        ge=200,
        le=750,
        description="Minimum VRAM (MB) yang harus tersedia sebelum loading model baru. "
                    "Ini adalah safety buffer untuk mencegah OOM (Out of Memory). "
                    "• 200-300: Untuk model kecil (<2GB). "
                    "• 400-500: Untuk model medium (2-7GB). "
                    "• 600-750: Untuk model besar (>7GB) atau multi-model setup."
    )

    vram_multiplier: float = Field(
        default=1.1,
        ge=1.0,
        le=3.0,
        description="Multiplier untuk estimasi VRAM yang dibutuhkan saat load model. "
                    "Formula: estimated_vram = file_size * vram_multiplier + kv_cache + overhead. "
                    "• 1.0-1.2: Model quantized (Q4, Q5). "
                    "• 1.3-1.5: Model semi-quantized (Q8). "
                    "• 1.5-2.0: Model FP16 atau context besar (>32K)."
    )

    keep_warm_models: int = Field(
        default=2,
        ge=0,
        description="Jumlah model yang tetap 'warm' (tidak di-unload meski idle). "
                    "Model dipilih berdasarkan popularitas (frekuensi request). "
                    "• 0: Semua model bisa di-unload. "
                    "• 1-2: Model utama tetap loaded. "
                    "• 3+: Untuk production dengan banyak model populer."
    )

    gpu_devices: list[int] = Field(
        default=[0],
        description="Daftar GPU device index yang akan digunakan. "
                    "Gunakan nvidia-smi untuk melihat GPU index. "
                    "• [0]: Single GPU (default). "
                    "• [0, 1]: Multi-GPU (untuk future support)."
    )

    parallel_requests: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Jumlah slot parallel request per model (llama.cpp --parallel). "
                    "Mempengaruhi berapa request yang bisa diproses bersamaan oleh satu model. "
                    "Trade-off: lebih tinggi = throughput lebih besar, tapi VRAM usage naik. "
                    "• 1-2: Model besar (>13B) atau context besar (>32K). "
                    "• 4-8: Model medium (7-13B). "
                    "• 8-16: Model kecil (<7B) dengan context kecil."
    )

    cpu_threads: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Jumlah CPU threads untuk operasi non-GPU (llama.cpp --threads). "
                    "Biasanya untuk tokenization dan post-processing. "
                    "Rekomendasi: setengah dari jumlah physical cores. "
                    "• 4-8: CPU 8-16 cores. "
                    "• 8-16: CPU 16-32 cores. "
                    "• 16+: Server dengan banyak cores."
    )

    use_mmap: bool = Field(
        default=True,
        description="Gunakan memory mapping untuk loading model file. "
                    "• True (default): Loading lebih cepat, model di-share di RAM. "
                    "• False: Loading lebih lambat, tapi lebih stabil untuk beberapa sistem. "
                    "Set False jika mengalami crash saat loading model besar."
    )

    flash_attention: str = Field(
        default="on",
        description="Flash Attention untuk efisiensi memory dan kecepatan. "
                    "• 'on': Aktifkan FA (recommended untuk GPU modern). "
                    "• 'off': Nonaktifkan FA (untuk GPU lama atau debugging). "
                    "Catatan: Memerlukan GPU yang support FA (NVIDIA Ampere+, AMD RDNA3+)."
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

    # HTTP Client Configuration
    http_max_keepalive: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maksimum keepalive connections untuk HTTP client ke llama-server"
    )

    http_max_connections: int = Field(
        default=200,
        ge=20,
        le=1000,
        description="Maksimum total connections untuk HTTP client ke llama-server"
    )

    # Queue Processor Configuration
    queue_processor_idle_sec: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Waktu idle (detik) sebelum queue processor berhenti. "
                    "Tingkatkan untuk workload dengan high latency."
    )

    # Model Loading Configuration
    model_load_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maksimum retry saat model gagal load. Set 0 untuk tidak retry."
    )

    @field_validator('llama_server_path')
    @classmethod
    def validate_llama_server_path(cls, v: str) -> str:
        # ENV variable LLAMA_SERVER_PATH takes precedence over config value
        # This is useful for Docker where the path is different from host
        env_path = os.getenv("LLAMA_SERVER_PATH", "")
        if env_path:
            v = env_path  # Override with ENV value

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

    @field_validator('base_models_path')
    @classmethod
    def validate_base_models_path(cls, v: str) -> str:
        # ENV variable BASE_MODELS_PATH takes precedence over config value
        # This is useful for Docker where the path is different from host
        env_path = os.getenv("BASE_MODELS_PATH", "")
        if env_path:
            v = env_path  # Override with ENV value

        # base_models_path is optional - if empty, all model_path must be absolute
        if v:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"base_models_path tidak ditemukan: {v}")
            if not path.is_dir():
                raise ValueError(f"base_models_path bukan directory: {v}")
        return v


class ModelParams(BaseModel):
    """
    Parameter spesifik per-model untuk llama-server.

    Parameter ini akan di-pass ke llama-server saat model di-load.
    Setiap model bisa memiliki konfigurasi berbeda sesuai kebutuhan.
    """

    n_gpu_layers: int = Field(
        default=99,
        ge=-1,
        description="Jumlah layer model yang di-load ke GPU. "
                    "• -1: Load semua layer ke GPU (full GPU offload). "
                    "• 0: CPU only (tidak menggunakan GPU). "
                    "• 99: Praktisnya sama dengan -1, load semua ke GPU. "
                    "• N: Load N layer pertama ke GPU, sisanya di CPU (hybrid)."
    )

    n_ctx: int = Field(
        default=4096,
        ge=512,
        le=131072,
        description="Context window size (jumlah token yang bisa diproses sekaligus). "
                    "Mempengaruhi VRAM usage secara signifikan. "
                    "• 2048-4096: Chat singkat, hemat VRAM. "
                    "• 8192-16384: Dokumen medium, RAG standard. "
                    "• 32768+: Dokumen panjang, RAG dengan banyak context. "
                    "Catatan: VRAM untuk KV cache ≈ n_ctx * 0.5MB (estimasi kasar)."
    )

    n_batch: int = Field(
        default=256,
        ge=128,
        le=512,
        description="Logical batch size untuk prompt processing. "
                    "Mempengaruhi kecepatan processing prompt panjang. "
                    "• 128-256: Untuk GPU dengan VRAM terbatas. "
                    "• 256-512: Untuk GPU dengan VRAM cukup. "
                    "Catatan: Nilai lebih tinggi = prompt processing lebih cepat."
    )

    rope_freq_base: Optional[int] = Field(
        default=None,
        ge=0,
        description="RoPE (Rotary Position Embedding) frequency base. "
                    "Untuk extended context length. "
                    "• None (default): Gunakan nilai default dari model. "
                    "• 10000: Default untuk kebanyakan model. "
                    "• 500000+: Untuk model dengan context extension (YaRN, etc)."
    )

    embedding: bool = Field(
        default=False,
        description="Aktifkan mode embedding untuk model ini. "
                    "• False: Model untuk text generation (chat, completion). "
                    "• True: Model untuk menghasilkan embeddings (RAG, semantic search). "
                    "Catatan: Gunakan model yang memang dirancang untuk embedding (BGE, Nomic, etc)."
    )

    chat_template: Optional[str] = Field(
        default=None,
        description="Override chat template untuk model ini. "
                    "• None: Gunakan template default dari model (recommended). "
                    "• 'chatml': Format ChatML. "
                    "• 'llama2': Format Llama 2. "
                    "• Custom Jinja template juga didukung."
    )

    parallel_override: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Override system.parallel_requests untuk model ini. "
                    "Berguna jika model tertentu butuh setting berbeda. "
                    "• None: Gunakan nilai dari system.parallel_requests. "
                    "• 1-2: Untuk model besar atau context panjang. "
                    "• 4+: Untuk model kecil dengan throughput tinggi."
    )

    batch_override: Optional[int] = Field(
        default=None,
        ge=128,
        le=4096,
        description="Override n_batch untuk model ini (llama.cpp --batch-size). "
                    "• None: Gunakan nilai n_batch default (256). "
                    "• 512-1024: Untuk prompt processing lebih cepat. "
                    "• 2048+: Untuk embedding models dengan input panjang."
    )

    type_k: Optional[str] = Field(
        default="f16",
        description="Tipe data untuk KV cache key. Mempengaruhi VRAM dan akurasi. "
                    "• 'f16' (default): Full precision, akurasi terbaik. "
                    "• 'q8_0': 8-bit quantized, hemat ~50% VRAM. "
                    "• 'q4_0': 4-bit quantized, hemat ~75% VRAM tapi akurasi turun. "
                    "Valid: f16, f32, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1"
    )

    type_v: Optional[str] = Field(
        default="f16",
        description="Tipe data untuk KV cache value. Sama seperti type_k. "
                    "Biasanya type_k dan type_v di-set sama. "
                    "Untuk context panjang (>32K), gunakan q4_0 atau q8_0 untuk hemat VRAM."
    )

    additional_parameter: Optional[str] = Field(
        default='',
        description="Additional raw llama.cpp-server command"
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
    """
    Konfigurasi per-model.

    Setiap model memiliki path file dan parameter spesifiknya sendiri.
    Model di-identifikasi dengan alias (key di dict models).
    """

    model_path: str = Field(
        ...,
        description="Path ke file model .gguf. "
                    "• Absolute path: '/home/user/models/model.gguf' "
                    "• Relative path: 'model.gguf' (relatif terhadap base_models_path). "
                    "File harus berformat GGUF dan accessible oleh server."
    )
    params: ModelParams = Field(
        default_factory=ModelParams,
        description="Parameter spesifik untuk model ini. "
                    "Lihat ModelParams untuk detail setiap parameter."
    )

    # Internal field to store resolved absolute path
    _resolved_path: Optional[str] = None

    def resolve_path(self, base_models_path: str) -> str:
        """
        Resolve model_path menjadi absolute path.
        - Jika model_path sudah absolute, langsung digunakan.
        - Jika relative, digabung dengan base_models_path.
        """
        path = Path(self.model_path)

        if path.is_absolute():
            resolved = path
        else:
            if not base_models_path:
                raise ValueError(
                    f"model_path '{self.model_path}' adalah relative path, "
                    f"tapi base_models_path tidak di-set di system config. "
                    f"Set base_models_path atau gunakan absolute path."
                )
            resolved = Path(base_models_path) / self.model_path

        # Validate resolved path
        if not resolved.exists():
            raise ValueError(f"Model file tidak ditemukan: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Model path bukan file: {resolved}")
        if not str(resolved).endswith('.gguf'):
            raise ValueError(f"Model file harus berformat .gguf: {resolved}")

        self._resolved_path = str(resolved)
        return self._resolved_path

    def get_resolved_path(self) -> str:
        """Get the resolved absolute path. Must call resolve_path first."""
        if self._resolved_path is None:
            raise RuntimeError(
                "resolve_path() belum dipanggil. Ini bug internal - "
                "seharusnya dipanggil saat AppConfig di-load."
            )
        return self._resolved_path


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

    @model_validator(mode='after')
    def resolve_all_model_paths(self) -> 'AppConfig':
        """
        Resolve semua model_path ke absolute path setelah config di-load.
        Ini memungkinkan model_path bisa relative terhadap base_models_path.
        """
        base_path = self.system.base_models_path

        for model_alias, model_conf in self.models.items():
            try:
                model_conf.resolve_path(base_path)
            except ValueError as e:
                raise ValueError(f"Error pada model '{model_alias}': {e}")

        return self


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
