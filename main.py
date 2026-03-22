import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

from ai_agent import (
    detect_crisis,
    detect_emotion,
    merge_session_after_turn,
    stream_user_input,
)
from murf_client import murf_client, generate_murf_voice, warmup as murf_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MEMORY_MAX_CHARS = 800
murf_semaphore = asyncio.Semaphore(3)


async def _tts_sentence(sentence: str):
    async with murf_semaphore:
        return await generate_murf_voice(sentence)


def _append_session_memory(prev: str, user_text: str, ai_text: str) -> str:
    line = f"User: {user_text[:180]} | Assistant: {ai_text[:180]}\n"
    combined = (prev + line).strip() if prev else line.strip()
    if len(combined) > MEMORY_MAX_CHARS:
        return combined[-MEMORY_MAX_CHARS:]
    return combined


def _cors_origins() -> list:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await murf_warmup()
    except Exception as e:
        logger.warning(f"Murf warmup failed (non-fatal): {e}")
    yield


app = FastAPI(
    title="Kalmly — Real-time wellness voice",
    description="Streaming LLM + ordered Murf TTS",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Kalmly Voice Gateway"}


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to primary WebSocket gateway.")

    session_messages: list = []
    session_memory = ""

    # STEP 1: Wrap the entire session in a persistent Murf stream
    # This ensures we pay the TLS handshake/auth cost ONLY ONCE 
    async with murf_client.get_session_stream() as murf_ws:
        try:
            while True:
                raw = await websocket.receive_text()

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"type": "text", "payload": raw}

                msg_type = data.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
                    continue

                if msg_type == "event":
                    logger.info(f"Event received: {data.get('payload')}")
                    continue

                user_text = data.get("payload", "") if msg_type == "text" else raw

                if not user_text.strip():
                    continue

                logger.info(f"Processing user input: {user_text}")

                emotion = detect_emotion(user_text)
                crisis = detect_crisis(user_text)

                await websocket.send_json({"type": "status", "content": "thinking"})

                sentence_queue: asyncio.Queue = asyncio.Queue()
                full_text_accum = ""

                # STEP 2: The Producer now uses the 'murf_ws' persistent stream
                async def produce():
                    try:
                        async for sentence in stream_user_input(user_text, session_messages, session_memory):
                            # Use synthesize_on_stream to reuse the open socket 
                            task = asyncio.create_task(
                                murf_client.synthesize_on_stream(murf_ws, sentence)
                            )
                            await sentence_queue.put((sentence, task))
                    except Exception as e:
                        logger.error(f"stream_user_input failed: {e}")
                        await websocket.send_json({"type": "error", "content": str(e)})
                    finally:
                        await sentence_queue.put(None)

                producer = asyncio.create_task(produce())

                try:
                    while True:
                        item = await sentence_queue.get()
                        if item is None:
                            break
                        
                        sentence, tts_task = item
                        
                        # STEP 3: Await the task results from the persistent stream
                        audio_bytes = await tts_task
                        
                        # Accumulate text for the UI
                        full_text_accum = (full_text_accum + " " + sentence).strip()
                        
                        # Send text (subtitles) followed immediately by bytes
                        await websocket.send_json({
                            "type": "text",
                            "content": full_text_accum,
                        })
                        
                        if audio_bytes:
                            await websocket.send_bytes(audio_bytes)
                            logger.info(f"Sent audio for sentence ({len(audio_bytes)} bytes)")
                finally:
                    await producer

                # Final metadata update for UI
                await websocket.send_json({
                    "type": "text",
                    "content": full_text_accum,
                    "emotion": emotion,
                    "crisis": crisis,
                })

                session_messages = merge_session_after_turn(session_messages, user_text, full_text_accum)
                session_memory = _append_session_memory(session_memory, user_text, full_text_accum)
                logger.info(f"Turn complete | emotion={emotion} crisis={crisis}")

        except WebSocketDisconnect:
            logger.info("Client disconnected normally.")
        except Exception as e:
            logger.error(f"WebSocket session error: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
