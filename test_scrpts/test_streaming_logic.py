
import os
from pathlib import Path

# Use local models-cache folder by default
base_path = Path(__file__).parent
os.environ["XDG_CACHE_HOME"] = os.getenv("XDG_CACHE_HOME", str(base_path / "models-cache"))

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import time
from streaming import StreamingTranscriber, StreamingResult
from config import config

class TestStreamingLogic(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_model = MagicMock()
        self.mock_vad = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_la = MagicMock()
        
        # Setup transcriber
        self.transcriber = StreamingTranscriber(
            model=self.mock_model,
            vad=self.mock_vad,
            processor=self.mock_processor,
            local_agreement=self.mock_la
        )
        
        # Default mocks behavior
        self.mock_processor.process_compressor.side_effect = lambda x: x
        self.mock_processor.process_noise_gate.side_effect = lambda x: x
        
        # Default VAD behavior: Always return chunks (Speech), Not Ended
        def vad_process_side_effect(audio):
            return [audio], False
        
        self.mock_vad.process.side_effect = vad_process_side_effect
        self.mock_vad.is_speaking = True # Simulate active speech
        
        # Default mock return for LA to avoid string += Mock
        default_la_result = MagicMock()
        default_la_result.confirmed_text = ""
        default_la_result.pending_text = ""
        default_la_result.full_text = ""
        self.mock_la.process.return_value = default_la_result
        
        # Config overrides for testing
        config.audio.sample_rate = 16000
        config.streaming.max_buffer_duration = 5.0 # Shorten for test
        config.streaming.forced_reset_overlap = 1.0
        config.streaming.min_silence_duration = 0.5
        config.streaming.inference_interval = 0.1
        
        # Disable Noise Removal to avoid dependency issues (DeepFilterNet)
        config.noise_removal.enabled = False


    def test_buffer_accumulation(self):
        """Test that audio adds to buffer."""
        self.transcriber.start()
        audio = np.zeros(16000, dtype=np.float32) # 1 sec
        self.transcriber.process(audio)
        self.assertEqual(len(self.transcriber._audio_buffer), 16000)

    def test_max_duration_forced_reset(self):
        """Test that buffer resets when exceeding max duration."""
        self.transcriber.start()
        
        # Max is 5.0s (80000 samples)
        # Add 4.0s
        audio_chunk = np.zeros(16000 * 4, dtype=np.float32)
        self.transcriber.process(audio_chunk)
        self.assertEqual(len(self.transcriber._audio_buffer), 64000)
        
        # Add 2.0s -> Total 6.0s > 5.0s -> Should trigger reset
        # The reset keeps 1.0s overlap
        audio_chunk_2 = np.zeros(16000 * 2, dtype=np.float32)
        
        # Setup mock return for the forced transcription
        self.mock_model.transcribe_audio.return_value.text = "forced commit text"
        self.mock_la.process.return_value.confirmed_text = "forced commit"
        self.mock_la.process.return_value.full_text = "forced commit"
        
        results = self.transcriber.process(audio_chunk_2)
        
        # Check buffer length: Should be roughly overlap size (1.0s = 16000)
        # Note: logic appends THEN checks (64000 + 32000 = 96000)
        # Reset -> keep last 16000
        self.assertEqual(len(self.transcriber._audio_buffer), 16000)
        
        # Check results
        self.assertTrue(len(results) > 0)
        self.assertTrue(results[0].is_final)
        self.assertIn("forced commit", self.transcriber._confirmed_text_history)

    def test_local_agreement_integration(self):
        """Test that LA is called and result used."""
        self.transcriber.start()
        self.transcriber._last_inference_time = 0 # Force run
        
        audio = np.zeros(16000, dtype=np.float32)
        
        self.mock_model.transcribe_audio.return_value.text = "hello world"
        self.mock_model.transcribe_audio.return_value.latency = 0.1
        
        # Mock LA returning stable "hello" and pending "world"
        la_result = MagicMock()
        la_result.confirmed_text = "hello"
        la_result.pending_text = "world"
        self.mock_la.process.return_value = la_result
        
        results = self.transcriber.process(audio)
        
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].confirmed_text, "hello")
        self.assertEqual(results[0].pending_text, "world")
        self.assertEqual(results[0].text, "hello world")
        
        # Verify confirmed history NOT updated (only updates on final)
        self.assertEqual(self.transcriber._confirmed_text_history, "")

if __name__ == '__main__':
    unittest.main()
