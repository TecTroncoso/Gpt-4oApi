from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, AsyncGenerator
import httpx
import asyncio
import json
from datetime import datetime
import urllib.parse

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
app = FastAPI(title="Pollinations.ai OpenAI Compatible API - Low Latency")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente HTTP asíncrono con configuración más robusta
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=httpx.Timeout(30.0),
    follow_redirects=True
)

# Almacenamiento en memoria
conversation_memory: Dict[str, List[Dict[str, str]]] = {}
MAX_MEMORY_LENGTH = 10

# Headers optimizados para evitar bloqueos
OPTIMIZED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0"
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "pollinations-ai"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150
    stream: Optional[bool] = False
    user: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

def clean_prompt(prompt: str) -> str:
    """Limpiar y codificar el prompt correctamente"""
    # Eliminar espacios extra y caracteres problemáticos
    cleaned = prompt.strip()
    # Codificar para URL
    return urllib.parse.quote(cleaned, safe='')

async def stream_response_generator(prompt: str, user_id: str) -> AsyncGenerator[str, None]:
    """Generador asíncrono para streaming de respuestas con manejo de errores"""
    try:
        # Limpiar y codificar el prompt
        cleaned_prompt = clean_prompt(prompt)
        url = f"https://text.pollinations.ai/{cleaned_prompt}"
        
        print(f"Requesting URL: {url}")  # Para debugging
        
        async with http_client.stream("GET", url, headers=OPTIMIZED_HEADERS) as response:
            print(f"Response status: {response.status_code}")  # Para debugging
            
            if response.status_code >= 400:
                error_msg = f"Server error '{response.status_code}' for url '{url}'"
                raise HTTPException(status_code=502, detail=error_msg)
            
            chunk_count = 0
            
            async for chunk in response.aiter_text(chunk_size=64):
                if chunk:
                    chunk_data = ChatCompletionChunk(
                        id=f"chatcmpl-{int(datetime.now().timestamp())}",
                        created=int(datetime.now().timestamp()),
                        model="pollinations-ai",
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta={"content": chunk},
                                finish_reason=None
                            )
                        ]
                    )
                    
                    yield f"data: {json.dumps(chunk_data.dict())}\n\n"
                    chunk_count += 1
                    
                    # Yield para mantener baja latencia
                    if chunk_count % 2 == 0:
                        await asyncio.sleep(0)
            
            # Chunk final
            final_chunk = ChatCompletionChunk(
                id=f"chatcmpl-{int(datetime.now().timestamp())}",
                created=int(datetime.now().timestamp()),
                model="pollinations-ai",
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
            
    except httpx.HTTPStatusError as e:
        error_data = {
            "error": {
                "message": f"HTTP Error: {str(e)}",
                "type": "http_error",
                "param": None,
                "code": e.response.status_code
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        error_data = {
            "error": {
                "message": f"Stream Error: {str(e)}",
                "type": "stream_error",
                "param": None,
                "code": None
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Construir el prompt con contexto de memoria
        user_id = request.user or "default"
        conversation_history = conversation_memory.get(user_id, [])
        
        # Obtener el último mensaje del usuario
        last_user_message = next((
            msg.content for msg in reversed(request.messages) 
            if msg.role == "user"
        ), "")
        
        # Construir el prompt (más simple para evitar problemas)
        prompt = last_user_message
        
        # Si se solicita streaming
        if request.stream:
            return StreamingResponse(
                stream_response_generator(prompt, user_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        
        # Para respuestas no-streaming
        else:
            cleaned_prompt = clean_prompt(prompt)
            url = f"https://text.pollinations.ai/{cleaned_prompt}"
            
            print(f"Non-stream request to: {url}")  # Para debugging
            
            try:
                response = await http_client.get(url, headers=OPTIMIZED_HEADERS)
                
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=502, 
                        detail=f"Server error '{response.status_code}' for url '{url}'"
                    )
                
                ai_response = response.text.strip()
                
                # Actualizar la memoria de conversación
                conversation_memory[user_id] = conversation_memory.get(user_id, [])[-(MAX_MEMORY_LENGTH-2):] + [
                    {"role": "user", "content": last_user_message},
                    {"role": "assistant", "content": ai_response}
                ]
                
                # Calcular tokens aproximados
                prompt_tokens = len(prompt.split())
                completion_tokens = len(ai_response.split())
                
                return ChatCompletionResponse(
                    id=f"chatcmpl-{int(datetime.now().timestamp())}",
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=ai_response),
                            finish_reason="stop"
                        )
                    ],
                    usage=ChatCompletionResponseUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                )
                
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Server error '{e.response.status_code}' for url '{url}'"
                )
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "pollinations-ai",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "pollinations"
            }
        ]
    }

@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    """Eliminar la memoria de conversación para un usuario específico"""
    if user_id in conversation_memory:
        del conversation_memory[user_id]
    return {"message": "Memory cleared"}

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Obtener la memoria de conversación para un usuario"""
    return {"conversation": conversation_memory.get(user_id, [])}

@app.get("/")
async def root():
    """Endpoint de health check"""
    return {"message": "Pollinations.ai API Proxy is running"}

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
