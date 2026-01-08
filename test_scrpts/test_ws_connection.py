import asyncio
import websockets
import sys

import ssl

async def test_connection():
    uri = "wss://inference.adeoaiengine.ecouncil.ae/models/5b1c1b58-f0e1-4ed1-aeb1-85ce3be73a14/ws/v1/audio/transcriptions"
    print(f"Attempting to connect to: {uri}")
    
    # Create an SSL context that disables verification
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print(f"\n✅ Connection established successfully!")
            
            print("Sending ping...")
            await websocket.ping()
            print("Ping sent successfully.")
            
            # Wait a moment to see if server closes or sends anything
            # await asyncio.sleep(1)
            
            print("Closing connection...")
            await websocket.close()
            print("Connection closed gracefully.")
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if websockets is installed
    try:
        import websockets
    except ImportError:
        print("Error: 'websockets' library is required.")
        print("Please run: pip install websockets")
        sys.exit(1)

    asyncio.run(test_connection())
