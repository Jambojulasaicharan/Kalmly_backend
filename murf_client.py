from __future__ import annotations

import os
import json
import base64
import logging
import websockets
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Security best practice: prioritize environment variables over hardcoded keys [cite: 112]
MURF_API_KEY = os.getenv("MURF_API_KEY")

class MurfStreamClient:
    """
    Manages high-fidelity TTS streaming via Murf AI.
    Updated to support persistent connections to minimize latency.
    """

    @asynccontextmanager
    async def get_session_stream(self):
        """
        Creates a single persistent WebSocket connection for the duration of a user session.
        This eliminates repeated TLS handshake and authentication overhead.
        """
        if not MURF_API_KEY:
            logger.error("MURF_API_KEY is missing.")
            raise ValueError("MURF_API_KEY not configured")

        ws_url = (
            f"wss://global.api.murf.ai/v1/speech/stream-input"
            f"?api-key={MURF_API_KEY}&model=FALCON&sample_rate=24000&channel_type=MONO&format=WAV"
        )

        async with websockets.connect(ws_url) as websocket:
            logger.info("Murf persistent stream: Connected.")
            
            # Send voice configuration ONCE at the start of the session 
            await websocket.send(
                json.dumps({
                    "voice_config": {
                        "voiceId": "Natalie",
                        "style": "Conversation",
                    }
                })
            )
            yield websocket
            logger.info("Murf persistent stream: Closed.")

    async def synthesize_on_stream(self, websocket, text_input: str):
        """
        Sends text to an ALREADY OPEN websocket and returns the binary audio.
        This allows for fluid transitions between sentences[cite: 58, 60].
        """
        if not (text_input or "").strip():
            return None

        try:
            # Send only the text and end flag to the existing stream [cite: 12]
            await websocket.send(
                json.dumps({
                    "text": text_input,
                    "end": True,
                })
            )

            audio_data = b""
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if "audio" in data:
                    audio_data += base64.b64decode(data["audio"])

                # Wait for the 'final' flag for this specific text chunk 
                if data.get("final"):
                    break

            return audio_data if audio_data else None

        except Exception as e:
            logger.error(f"Murf streaming synthesis error: {e}")
            return None

    # Legacy method kept for simple standalone calls if needed 
    async def synthesize(self, text_input: str):
        async with self.get_session_stream() as ws:
            return await self.synthesize_on_stream(ws, text_input)

    async def warmup(self) -> None:
        """Primes the network path on server startup[cite: 16, 17]."""
        logger.info("Murf warmup: sending dummy synthesis…")
        await self.synthesize("Hello")

# Single shared instance [cite: 18]
murf_client = MurfStreamClient()

async def generate_murf_voice(text_input: str) -> bytes | None:
    return await murf_client.synthesize(text_input)

async def warmup() -> None:
    await murf_client.warmup()