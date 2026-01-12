
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
        
        self.compressor_enabled = config.compressor.enabled
        self.noise_gate_enabled = config.noise_gate.enabled
        
        # Initialize Compressor
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
        if not self.compressor_enabled:
            return audio
            
        return self.compressor.process(audio, sample_rate=self.sample_rate)

    def process_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gating."""
        if not self.noise_gate_enabled:
            return audio
            
        return self.noise_gate.process(audio, sample_rate=self.sample_rate)
        
    @property
    def compressor_threshold(self):
        return self.compressor.threshold_db
        
    @compressor_threshold.setter
    def compressor_threshold(self, value):
        self.compressor.threshold_db = value
        
    @property
    def compressor_ratio(self):
        return self.compressor.ratio
        
    @compressor_ratio.setter
    def compressor_ratio(self, value):
        self.compressor.ratio = value
        
    @property
    def noise_gate_threshold(self):
        return self.noise_gate.threshold_db
        
    @noise_gate_threshold.setter
    def noise_gate_threshold(self, value):
        self.noise_gate.threshold_db = value
