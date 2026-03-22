import asyncio
import websockets
import json
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("MURF_API_KEY")
print(f"Testing key: {key[:12]}...")

async def test():
    url = f"wss://global.api.murf.ai/v1/speech/stream-input?api-key=ap2_8fc9310d-8e69-4a3b-98c1-f7f77854ccc1&model=FALCON&sample_rate=24000&channel_type=MONO&format=WAV"
    try:
        async with websockets.connect(url) as ws:
            print("Connected OK!")
            await ws.send(json.dumps({"voice_config": {"voiceId": "Natalie", "style": "Conversation"}}))
            await ws.send(json.dumps({"text": "Hello", "end": True}))
            resp = await ws.recv()
            print("Response:", resp[:200])
    except Exception as e:
        print("Error:", e)

asyncio.run(test())