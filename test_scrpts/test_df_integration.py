
import os
import numpy as np
import audio_buffer
from streaming import StreamingTranscriber
from config import config

def test_df_integration():
    print("Testing DeepFilterNet Integration in StreamingTranscriber...")
    
    # Ensure model cache env var if not set (for local testing)
    if "XDG_CACHE_HOME" not in os.environ:
         base_path = os.path.dirname(os.path.abspath(__file__))
         os.environ["XDG_CACHE_HOME"] = os.path.join(base_path, "models-cache")

    # Initialize transcriber
    transcriber = StreamingTranscriber()
    
    # Generate some dummy audio (white noise)
    # 1 second of audio at 16k
    sr = 16000
    duration = 1.0
    audio = np.random.uniform(-0.5, 0.5, int(sr * duration)).astype(np.float32)
    
    print("Starting transcriber...")
    transcriber.start()
    
    # Check if DF state is initialized
    if transcriber.df_state is None:
        print("ERROR: DF state not initialized!")
        return False
    else:
        print(f"DF state initialized: {type(transcriber.df_state)}")

    print("Processing audio chunk...")
    try:
        results = transcriber.process(audio)
        print(f"Process successful. Got {len(results)} results.")
    except Exception as e:
        print(f"ERROR during process: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("Test passed!")
    return True

if __name__ == "__main__":
    test_df_integration()
