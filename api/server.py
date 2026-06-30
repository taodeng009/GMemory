import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

load_dotenv()
os.environ.setdefault("OPENAI_API_BASE", "")
os.environ.setdefault("OPENAI_API_KEY", "")

from .projector import ProjectorService
from .schemas import (
    EpisodeRequest,
    HealthResponse,
    ProjectorRequest,
    ProjectorResponse,
    RetrieveRequest,
)
from .service import GMemoryApiService


app = FastAPI(title="GMemory API", version="0.1.0")
service = GMemoryApiService()


class _LazyProjectorLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._client = None

    def __call__(self, *args, **kwargs):
        if self._client is None:
            from mas.llm import GPTChat

            self._client = GPTChat(model_name=self.model_name)
        return self._client(*args, **kwargs)


projector_service = ProjectorService(
    llm_client=_LazyProjectorLLM(model_name=service.config.llm_model),
    tracer=service.tracer,
)


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


@app.post("/api/v1/memory/project", response_model=ProjectorResponse)
def project_insights(request: ProjectorRequest):
    return projector_service.project(request)


@app.post("/api/v1/memory/episodes")
def save_episode(request: EpisodeRequest):
    return service.save_episode(request)
