import json
import time
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Literal
import httpx
import uuid

app = FastAPI()

# OpenAI-style models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ollama"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]

# Legacy endpoints (keeping for backward compatibility)
class Query(BaseModel):
    prompt: str
    model: str = "llama2"
    stream: Optional[bool] = True

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI chat messages to a single prompt string"""
    prompt_parts = []
    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}")
        elif message.role == "user":
            prompt_parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")
    
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

# OpenAI-style streaming response generator
async def stream_openai_chat_completion(messages: List[ChatMessage], model: str):
    prompt = messages_to_prompt(messages)
    url = "http://localhost:11434/api/generate"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST", url, json={"model": model, "prompt": prompt}
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to connect to the model server."
                    )

                chunk_id = str(uuid.uuid4())
                async for chunk in response.aiter_bytes():
                    decoded_chunk = chunk.decode('utf-8')
                    for line in decoded_chunk.split("\n\n"):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data and data["response"]:
                                    # OpenAI-style streaming format
                                    chunk_response = {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": data["response"]
                                            },
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_response)}\n\n"
                                
                                if data.get("done", False):
                                    # Final chunk
                                    final_chunk = {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop"
                                        }]
                                    }
                                    yield f"data: {json.dumps(final_chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                            except json.JSONDecodeError:
                                continue

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

# OpenAI-style completion streaming
async def stream_openai_completion(prompt: str, model: str):
    url = "http://localhost:11434/api/generate"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST", url, json={"model": model, "prompt": prompt}
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to connect to the model server."
                    )

                completion_id = str(uuid.uuid4())
                async for chunk in response.aiter_bytes():
                    decoded_chunk = chunk.decode('utf-8')
                    for line in decoded_chunk.split("\n\n"):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data and data["response"]:
                                    chunk_response = {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "text": data["response"],
                                            "index": 0,
                                            "logprobs": None,
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_response)}\n\n"
                                
                                if data.get("done", False):
                                    final_chunk = {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "text": "",
                                            "index": 0,
                                            "logprobs": None,
                                            "finish_reason": "stop"
                                        }]
                                    }
                                    yield f"data: {json.dumps(final_chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                            except json.JSONDecodeError:
                                continue

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

# OpenAI-style non-streaming chat completion
async def get_openai_chat_completion(messages: List[ChatMessage], model: str):
    prompt = messages_to_prompt(messages)
    url = "http://localhost:11434/api/generate"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json={"model": model, "prompt": prompt})
            response.raise_for_status()

            combined_response = ""
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            combined_response += data["response"]
                    except json.JSONDecodeError:
                        continue

            return {
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": combined_response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(combined_response.split()),
                    "total_tokens": len(prompt.split()) + len(combined_response.split())
                }
            }

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

# OpenAI-style non-streaming completion
async def get_openai_completion(prompt: str, model: str):
    url = "http://localhost:11434/api/generate"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json={"model": model, "prompt": prompt})
            response.raise_for_status()

            combined_response = ""
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            combined_response += data["response"]
                    except json.JSONDecodeError:
                        continue

            return {
                "id": str(uuid.uuid4()),
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "text": combined_response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(combined_response.split()),
                    "total_tokens": len(prompt.split()) + len(combined_response.split())
                }
            }

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

# OpenAI-compatible endpoints
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_openai_chat_completion(request.messages, request.model),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        response = await get_openai_chat_completion(request.messages, request.model)
        return JSONResponse(response)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    
    if request.stream:
        return StreamingResponse(
            stream_openai_completion(prompt, request.model),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        response = await get_openai_completion(prompt, request.model)
        return JSONResponse(response)

@app.get("/v1/models")
async def list_openai_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            
            ollama_models = response.json()["models"]
            openai_models = []
            
            for model in ollama_models:
                openai_models.append(Model(
                    id=model["name"],
                    created=int(time.time())
                ))
            
            return ModelsResponse(data=openai_models)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# Legacy endpoints (keeping for backward compatibility)
async def stream_generated_text(prompt: str, model: str):
    url = "http://localhost:11434/api/generate"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST", url, json={"model": model, "prompt": prompt}
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to connect to the model server."
                    )

                async for chunk in response.aiter_bytes():
                    decoded_chunk = chunk.decode('utf-8')
                    for line in decoded_chunk.split("\n\n"):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

async def get_generated_text(prompt: str, model: str):
    url = "http://localhost:11434/api/generate"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json={"model": model, "prompt": prompt})
            response.raise_for_status()

            combined_response = ""
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            combined_response += data["response"]
                    except json.JSONDecodeError:
                        continue

            return {"response": combined_response}

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

@app.post("/api/generate")
async def generate_text(query: Query):
    if query.stream:
        return StreamingResponse(
            stream_generated_text(query.prompt, query.model),
            media_type="text/plain"
        )
    else:
        response = await get_generated_text(query.prompt, query.model)
        return JSONResponse(response)

@app.post("/api/models/download")
async def download_model(llm_name: str = Body(..., embed=True)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/pull",
                json={"name": llm_name}
            )
            response.raise_for_status()
            return {"message": f"Model {llm_name} downloaded successfully"}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            return {"models": response.json()["models"]}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)