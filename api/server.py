from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

load_dotenv()

from .schemas import EpisodeRequest, HealthResponse, RetrieveRequest
from .service import GMemoryApiService


app = FastAPI(title="GMemory API", version="0.1.0")
service = GMemoryApiService()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = service.tracer.new_trace_id()
    errors = jsonable_encoder(exc.errors())
    error = f"RequestValidationError: {errors}"
    response = {"detail": errors, "trace_id": trace_id, "error": error}
    try:
        body = await request.json()
    except Exception:
        body = {}
    service.tracer.record(
        trace_id,
        request.url.path,
        body,
        {"validation_error": True},
        response,
        error,
    )
    return JSONResponse(status_code=422, content=response)


@app.get("/api/v1/memory/health", response_model=HealthResponse)
def health():
    return service.health()


@app.post("/api/v1/memory/retrieve")
def retrieve_memory(request: RetrieveRequest):
    return service.retrieve(request)


@app.post("/api/v1/memory/episodes")
def save_episode(request: EpisodeRequest):
    return service.save_episode(request)
