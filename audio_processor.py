
import numpy as np
from pedalboard import Compressor, NoiseGate
from config import config

class AudioProcessor:
    """
    Handles audio processing effects using Pedalboard.
    Stateful plugins (Compressor, NoiseGate) are maintained here.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Initialize Compressor
        # We initialize with current config. 
        # Note: If config changes at runtime, we might need a method to update these.
        # For now assuming static config per session.
        self.compressor = Compressor(
            threshold_db=config.compressor.threshold_db,
            ratio=config.compressor.ratio,
            attack_ms=config.compressor.attack_ms,
            release_ms=config.compressor.release_ms
        )
        
        # Initialize Noise Gate
        self.noise_gate = NoiseGate(
            threshold_db=config.noise_gate.threshold_db,
            ratio=10.0,
            attack_ms=1.0,
            release_ms=100.0
        )

    def process_compressor(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        if not config.compressor.enabled:
            return audio
            
        # Pedalboard process returns the processed audio
        # It handles state internally
        return self.compressor.process(audio, sample_rate=self.sample_rate)

    def process_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gating."""
        if not config.noise_gate.enabled:
            return audio
            
        return self.noise_gate.process(audio, sample_rate=self.sample_rate)
