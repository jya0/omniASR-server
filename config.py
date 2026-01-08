"""Configuration settings for omniASR streaming server."""

import os
from dataclasses import dataclass, field
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use env vars directly


def env_str(key: str, default: str) -> str:
    """Get string from environment."""
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    """Get int from environment."""
    return int(os.environ.get(key, default))


def env_float(key: str, default: float) -> float:
    """Get float from environment."""
    return float(os.environ.get(key, default))


def env_bool(key: str, default: bool) -> bool:
    """Get bool from environment."""
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


def env_optional_str(key: str, default: Optional[str]) -> Optional[str]:
    """Get optional string from environment."""
    val = os.environ.get(key)
    if val is None:
        return default
    if val.lower() in ("none", "null", ""):
        return None
    return val


@dataclass
class AudioConfig:
    """Audio processing settings."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"  # For WebSocket raw PCM


@dataclass
class StreamingConfig:
    """Streaming and chunking settings."""

    chunk_duration: float = 5.0  # seconds per processing chunk
    overlap_ratio: float = 0.5  # overlap between chunks (0.5 = 50%)
    min_chunk_duration: float = 0.3  # minimum audio to process
    max_buffer_duration: float = 30.0  # max buffer before forced flush


@dataclass
class LocalAgreementConfig:
    """LocalAgreement algorithm settings."""

    min_agreement: int = 2  # chunks must agree before confirming
    prefix_match_ratio: float = 0.8  # how similar prefixes must be to "agree"


@dataclass
class TenVADConfig:
    """TenVAD settings."""
    
    hop_size: int = 256  # 16 ms per frame (fixed by model)
    threshold: float = 0.7  # Detection threshold (Higher = More aggressive/Stricter)


@dataclass
class VADConfig:
    """Voice Activity Detection settings."""

    enabled: bool = False  # Set False when using external VAD (Pipecat, LiveKit)
    type: str = "ten_vad"  # "rms" or "ten_vad"
    silence_threshold: float = 0.03  # RMS threshold (Higher = Cuts louder noise)
    silence_duration: float = 2.0  # seconds of silence = end of utterance
    min_speech_duration: float = 0.1  # minimum speech to process
    ten_vad: TenVADConfig = field(default_factory=TenVADConfig)


@dataclass
class NoiseRemovalConfig:
    """Noise removal settings (DeepFilterNet)."""
    
    enabled: bool = False
    attenuation: float = 0.5  # 1.0 = Full removal, 0.0 = No removal (Dry/Wet mix)


@dataclass
class ModelConfig:
    """Model settings."""

    model_card: str = field(default_factory=lambda: env_str("MODEL_CARD", "omniASR_CTC_300M_v2"))
    default_lang: str | None = field(default_factory=lambda: env_optional_str("DEFAULT_LANG", "eng_Latn"))
    batch_size: int = field(default_factory=lambda: env_int("BATCH_SIZE", 1))
    device: str | None = field(default_factory=lambda: env_optional_str("DEVICE", None))  # None = auto-detect


@dataclass
class ServerConfig:
    """Server settings."""

    host: str = field(default_factory=lambda: env_str("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: env_int("PORT", 8080))
    cors_origins: list[str] | None = None
    # Concurrent session limits
    max_concurrent_requests: int = field(default_factory=lambda: env_int("MAX_CONCURRENT_REQUESTS", 100))
    max_websocket_connections: int = field(default_factory=lambda: env_int("MAX_WEBSOCKET_CONNECTIONS", 50))


@dataclass
class StreamingEnvConfig:
    """Streaming settings from environment."""

    chunk_duration: float = field(default_factory=lambda: env_float("CHUNK_DURATION", 5.0))
    vad_enabled: bool = field(default_factory=lambda: env_bool("VAD_ENABLED", False))
    noise_removal_enabled: bool = field(default_factory=lambda: env_bool("NOISE_REMOVAL_ENABLED", False))


@dataclass
class Config:
    """Main configuration container."""

    audio: AudioConfig
    streaming: StreamingConfig
    local_agreement: LocalAgreementConfig
    vad: VADConfig
    noise_removal: NoiseRemovalConfig
    model: ModelConfig
    server: ServerConfig

    @classmethod
    def default(cls) -> "Config":
        # Get env overrides
        env_config = StreamingEnvConfig()

        streaming = StreamingConfig()
        streaming.chunk_duration = env_config.chunk_duration

        vad = VADConfig()
        vad.enabled = env_config.vad_enabled
        
        noise_removal = NoiseRemovalConfig()
        noise_removal.enabled = env_config.noise_removal_enabled

        return cls(
            audio=AudioConfig(),
            streaming=streaming,
            local_agreement=LocalAgreementConfig(),
            vad=VADConfig(ten_vad=TenVADConfig()),
            noise_removal=noise_removal,
            model=ModelConfig(),
            server=ServerConfig(),
        )


# Global default config
config = Config.default()
