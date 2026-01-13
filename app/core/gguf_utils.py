"""
GGUF Metadata Parser untuk Model Info dan SWA Detection

Modul ini menyediakan parser untuk membaca metadata dari file model GGUF
tanpa memerlukan library gguf resmi. Digunakan untuk:
- Mendeteksi apakah model menggunakan Sliding Window Attention (SWA)
- Membaca informasi arsitektur model
- Menentukan optimal parallel slots berdasarkan context window

GGUF Format:
    GGUF adalah format file model untuk llama.cpp yang menyimpan:
    - Magic number: 'GGUF' (4 bytes)
    - Version (uint32)
    - Tensor count (uint64)
    - Metadata KV count (uint64)
    - Metadata key-value pairs
    - Tensor data

Components:
    - GGUFModelInfo: Dataclass dengan info model terstruktur
    - GGUFReader: Parser minimal untuk read GGUF header dan metadata
    - detect_swa(): Quick check untuk SWA support
    - get_model_info(): Get full model info
    - get_optimal_parallel(): Calculate optimal parallel slots

SWA (Sliding Window Attention):
    Model dengan SWA (seperti Gemma, Mistral) menggunakan attention window
    yang terbatas, bukan full attention. Ini mempengaruhi:
    - Context handling: Model hanya "melihat" token dalam window
    - Context shifting: Perlu diaktifkan untuk conversations panjang
    - Parallel slots: Perlu konteks yang cukup per slot

Usage:
    # Quick SWA check
    is_swa, window_size = detect_swa("/path/to/model.gguf")
    
    # Full model info
    info = get_model_info("/path/to/model.gguf")
    print(f"Architecture: {info.architecture}")
    print(f"SWA: {info.is_swa}, Window: {info.swa_window_size}")
    
    # Optimal parallel calculation
    parallel, reason = get_optimal_parallel(
        model_path="/path/to/model.gguf",
        n_ctx=8192,
        default_parallel=4
    )
"""

import struct
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# GGUF Magic Number
GGUF_MAGIC = b'GGUF'

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class GGUFModelInfo:
    """Container untuk informasi model dari GGUF metadata."""
    architecture: str = "unknown"
    name: str = "unknown"
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0

    # SWA (Sliding Window Attention) info
    is_swa: bool = False
    swa_window_size: int = 0

    # Additional info
    vocab_size: int = 0

    # Raw metadata for debugging
    raw_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_metadata is None:
            self.raw_metadata = {}


class GGUFReader:
    """
    Minimal GGUF reader untuk extract header dan metadata.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.metadata: Dict[str, Any] = {}
        self.version: int = 0
        self.tensor_count: int = 0
        self.metadata_kv_count: int = 0

    def _read_string(self, f) -> str:
        """Read GGUF string (length-prefixed)."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, value_type: int) -> Any:
        """Read value based on type."""
        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUF_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUF_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUF_TYPE_BOOL:
            return struct.unpack('<B', f.read(1))[0] != 0
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        elif value_type == GGUF_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF value type: {value_type}")

    def read_metadata(self) -> Dict[str, Any]:
        """
        Read all metadata from GGUF file.
        Returns dict of key-value pairs.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")

        with open(self.file_path, 'rb') as f:
            # Read magic
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Invalid GGUF file: magic={magic}, expected={GGUF_MAGIC}")

            # Read version
            self.version = struct.unpack('<I', f.read(4))[0]
            if self.version < 2:
                raise ValueError(f"Unsupported GGUF version: {self.version}")

            # Read counts
            self.tensor_count = struct.unpack('<Q', f.read(8))[0]
            self.metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(self.metadata_kv_count):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

        return self.metadata

    def get_model_info(self) -> GGUFModelInfo:
        """
        Parse metadata dan return structured model info.
        """
        if not self.metadata:
            self.read_metadata()

        info = GGUFModelInfo()
        info.raw_metadata = self.metadata.copy()

        # Get architecture
        info.architecture = self.metadata.get(
            'general.architecture', 'unknown')
        info.name = self.metadata.get('general.name', 'unknown')

        # Get context length (try multiple keys)
        arch = info.architecture
        info.context_length = (
            self.metadata.get(f'{arch}.context_length', 0) or
            self.metadata.get('llama.context_length', 0) or
            self.metadata.get('general.context_length', 0)
        )

        # Get embedding/hidden size
        info.embedding_length = (
            self.metadata.get(f'{arch}.embedding_length', 0) or
            self.metadata.get('llama.embedding_length', 0)
        )

        # Get block/layer count
        info.block_count = (
            self.metadata.get(f'{arch}.block_count', 0) or
            self.metadata.get('llama.block_count', 0)
        )

        # Detect SWA (Sliding Window Attention)
        # Different models use different key names
        swa_keys = [
            f'{arch}.attention.sliding_window',
            f'{arch}.attention.sliding_window_size',
            'llama.attention.sliding_window',
            'mistral.attention.sliding_window',
            'gemma.attention.sliding_window',
            'gemma2.attention.sliding_window',
            'gemma3.attention.sliding_window',
        ]

        for key in swa_keys:
            if key in self.metadata:
                swa_value = self.metadata[key]
                if isinstance(swa_value, (int, float)) and swa_value > 0:
                    info.is_swa = True
                    info.swa_window_size = int(swa_value)
                    break

        # Get vocab size
        info.vocab_size = self.metadata.get(f'{arch}.vocab_size', 0)

        return info


def detect_swa(model_path: str) -> Tuple[bool, int]:
    """
    Quick function to detect if model uses SWA.

    Args:
        model_path: Path to GGUF model file

    Returns:
        Tuple of (is_swa, window_size)
        - is_swa: True if model uses Sliding Window Attention
        - window_size: Size of sliding window (0 if not SWA)
    """
    try:
        reader = GGUFReader(model_path)
        info = reader.get_model_info()
        return info.is_swa, info.swa_window_size
    except Exception as e:
        logger.warning(f"Failed to detect SWA for {model_path}: {e}")
        return False, 0


def get_model_info(model_path: str) -> Optional[GGUFModelInfo]:
    """
    Get full model info from GGUF file.

    Args:
        model_path: Path to GGUF model file

    Returns:
        GGUFModelInfo or None if failed
    """
    try:
        reader = GGUFReader(model_path)
        return reader.get_model_info()
    except Exception as e:
        logger.warning(f"Failed to read GGUF metadata for {model_path}: {e}")
        return None


def get_optimal_parallel(
    model_path: str,
    n_ctx: int,
    default_parallel: int,
    min_ctx_per_slot: int = 2048
) -> Tuple[int, str]:
    """
    Determine optimal parallel setting based on model characteristics.

    Args:
        model_path: Path to GGUF model file
        n_ctx: Total context size
        default_parallel: Default parallel value from config
        min_ctx_per_slot: Minimum context tokens per slot

    Returns:
        Tuple of (optimal_parallel, reason)
    """
    try:
        reader = GGUFReader(model_path)
        info = reader.get_model_info()

        if info.is_swa:
            # For SWA models, we need more context per slot
            # to allow proper context shifting
            swa_window = info.swa_window_size

            # Minimum context per slot should be at least 4x SWA window
            # to allow meaningful context shifting
            min_ctx_for_swa = max(min_ctx_per_slot, swa_window * 4)

            # Calculate optimal parallel
            optimal = max(1, n_ctx // min_ctx_for_swa)

            if optimal < default_parallel:
                reason = (
                    f"SWA model detected (window={swa_window}). "
                    f"Reduced parallel from {default_parallel} to {optimal} "
                    f"(ctx/slot: {n_ctx // optimal} tokens)"
                )
                return optimal, reason
            else:
                reason = f"SWA model (window={swa_window}), parallel={default_parallel} is OK"
                return default_parallel, reason
        else:
            # Non-SWA model: still apply minimum context per slot
            effective_ctx = n_ctx // default_parallel

            if effective_ctx < min_ctx_per_slot and default_parallel > 1:
                optimal = max(1, n_ctx // min_ctx_per_slot)
                reason = (
                    f"Non-SWA model, but context per slot too small. "
                    f"Adjusted parallel from {default_parallel} to {optimal}"
                )
                return optimal, reason
            else:
                reason = f"Non-SWA model, parallel={default_parallel} is OK"
                return default_parallel, reason

    except Exception as e:
        logger.warning(
            f"Failed to analyze model {model_path}: {e}. Using default parallel={default_parallel}")
        return default_parallel, f"Analysis failed: {e}"
