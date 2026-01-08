import sys
import os
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from audio_buffer import VAD

def test_ten_vad_integration():
    print("Testing TenVAD Integration...")
    
    # Ensure config is set to use ten_vad
    if config.vad.type != "ten_vad":
        print(f"WARNING: Configured VAD type is {config.vad.type}, forcing 'ten_vad' for test.")
        config.vad.type = "ten_vad"
        
    vad = VAD()
    
    if vad.ten_vad is None:
        print("ERROR: TenVAD failed to initialize.")
        return False
        
    print("TenVAD Initialized successfully.")
    
    # Test Silence (0.5s)
    print("\nTesting Silence...")
    silence = np.zeros(int(16000 * 0.5), dtype=np.float32)
    is_speech, speech_ended = vad.process(silence)
    print(f"Silence Result: is_speech={is_speech}, speech_ended={speech_ended}")
    
    if is_speech:
        print("ERROR: Silence detected as speech!")
        return False
        
    # Test Signal (0.5s sine wave)
    print("\nTesting Signal (Sine Wave)...")
    t = np.linspace(0, 0.5, int(16000 * 0.5))
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    is_speech, speech_ended = vad.process(signal)
    print(f"Signal Result: is_speech={is_speech}, speech_ended={speech_ended}")
    
    if not is_speech:
        print("ERROR: Signal NOT detected as speech!")
        return False
        
    print("\nSUCCESS: TenVAD integration verified.")
    return True

if __name__ == "__main__":
    success = test_ten_vad_integration()
    sys.exit(0 if success else 1)
