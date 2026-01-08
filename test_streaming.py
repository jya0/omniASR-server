"""
Test script for omniASR streaming pipeline.

Tests:
1. Microphone streaming with real-time output
2. WebSocket client test
"""

import os

from pathlib import Path

# Use local models-cache folder by default
base_path = Path(__file__).parent
os.environ["XDG_CACHE_HOME"] = os.getenv("XDG_CACHE_HOME", str(base_path / "models-cache"))

import argparse
import asyncio
import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from config import config


def test_microphone_streaming():
    """Test streaming with microphone input."""
    from streaming import StreamingTranscriber

    print("=" * 60)
    print("omniASR Streaming Test - Microphone Input")
    print("=" * 60)
    print()

    transcriber = StreamingTranscriber()

    def on_result(result):
        if result.is_final:
            print(f"\n>> FINAL: {result.text}")
            print(f"   [latency: {result.latency_ms:.0f}ms | audio: {result.audio_duration:.1f}s]")
            print()
        else:
            # Show confirmed in green, pending in gray
            confirmed = result.confirmed_text
            pending = result.pending_text
            print(f"\r[{confirmed}] {pending}...", end="", flush=True)

    transcriber.on_result = on_result

    # Audio queue for thread communication
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        audio_queue.put(indata.copy())

    print("Starting... Press Ctrl+C to stop")
    print("[Speak into microphone]")
    print()

    transcriber.start()

    # Start audio stream
    chunk_size = int(config.audio.sample_rate * 0.1)  # 100ms chunks

    with sd.InputStream(
        samplerate=config.audio.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_size,
        callback=audio_callback,
    ):
        try:
            while True:
                try:
                    audio = audio_queue.get(timeout=0.1)
                    transcriber.process(audio.flatten())
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass

    # End session
    final = transcriber.end()
    if final:
        print(f"\n>> SESSION END: {final.text}")

    print("\nDone.")


async def test_websocket_client(host: str = "localhost", port: int = 8000):
    """Test WebSocket streaming client."""
    import websockets
    import sys

    print("=" * 60)
    print("omniASR WebSocket Client Test")
    print("=" * 60)
    print()

    uri = f"ws://{host}:{port}/v1/audio/transcriptions"
    print(f"Connecting to {uri}...")

    audio_queue = queue.Queue()  # Use thread-safe queue

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    try:
        async with websockets.connect(uri) as ws:
            # Wait for ready message
            ready = await ws.recv()
            print(f"Server: {ready}")
            print()

            print("Starting... Press Ctrl+C to stop")
            print("[Speak into microphone]")
            print()

            # Start audio stream
            chunk_size = int(config.audio.sample_rate * 0.1)

            stream = sd.InputStream(
                samplerate=config.audio.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
                callback=audio_callback,
            )
            stream.start()

            async def receive_loop():
                """Receive and display results."""
                try:
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("type") == "error":
                            print(f"\nError: {data.get('message')}", flush=True)
                        elif "text" in data:
                            if data.get("is_final"):
                                sys.stdout.write(f"\r\033[K")  # Clear line
                                print(f">> FINAL: {data['text']}", flush=True)
                                print(f"   [latency: {data.get('latency_ms', 0):.0f}ms]\n", flush=True)
                            else:
                                confirmed = data.get("confirmed_text", "")
                                pending = data.get("pending_text", "")
                                sys.stdout.write(f"\r\033[K[{confirmed}] {pending}...")
                                sys.stdout.flush()
                except websockets.exceptions.ConnectionClosed:
                    pass

            async def send_loop():
                """Send audio chunks."""
                try:
                    while True:
                        # Non-blocking check for audio
                        try:
                            audio = audio_queue.get_nowait()
                            await ws.send(audio.tobytes())
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"\nSend error: {e}", flush=True)

            # Run both loops
            receiver = asyncio.create_task(receive_loop())
            sender = asyncio.create_task(send_loop())

            try:
                # Wait for either to complete (they shouldn't unless error)
                done, pending = await asyncio.wait(
                    [receiver, sender],
                    return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                pass
            finally:
                stream.stop()
                stream.close()
                receiver.cancel()
                sender.cancel()

                # Try to send end signal
                try:
                    await ws.send(json.dumps({"type": "end"}))
                except:
                    pass

    except websockets.exceptions.ConnectionRefused:
        print(f"\nError: Could not connect to {uri}")
        print("Make sure the server is running: python server.py")
    except Exception as e:
        print(f"\nError: {e}")

    print("\nDone.")


def test_rest_api(host: str = "localhost", port: int = 8000, audio_file: str = None):
    """Test REST API with audio file."""
    import requests

    print("=" * 60)
    print("omniASR REST API Test")
    print("=" * 60)
    print()

    base_url = f"http://{host}:{port}"

    # Health check
    print("Checking health...")
    resp = requests.get(f"{base_url}/health")
    print(f"Health: {resp.json()}")
    print()

    if not audio_file:
        print("No audio file provided. Use --file to test transcription.")
        return

    # Transcribe
    print(f"Transcribing: {audio_file}")
    with open(audio_file, "rb") as f:
        resp = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": f},
            data={"response_format": "verbose_json"},
        )

    if resp.status_code == 200:
        result = resp.json()
        print(f"\nResult: {result['text']}")
        print(f"Duration: {result.get('duration', 'N/A')}s")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="Test omniASR streaming")
    parser.add_argument(
        "mode",
        choices=["mic", "websocket", "rest"],
        help="Test mode: mic (direct streaming), websocket (via server), rest (file upload)",
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--file", help="Audio file for REST test")

    args = parser.parse_args()

    try:
        if args.mode == "mic":
            test_microphone_streaming()
        elif args.mode == "websocket":
            asyncio.run(test_websocket_client(args.host, args.port))
        elif args.mode == "rest":
            test_rest_api(args.host, args.port, args.file)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")


if __name__ == "__main__":
    main()
