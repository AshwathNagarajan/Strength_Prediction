import json

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import datasets, health, models, optimization, prediction, reports, training
from app.core.logging import configure_logging
from app.core.config import settings

configure_logging()

app = FastAPI(title=settings.app_name, version="2.0.0", debug=settings.debug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for router in (health.router, datasets.router, training.router, models.router, prediction.router, optimization.router, reports.router):
    app.include_router(router)


@app.middleware("http")
async def consistent_api_response(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if not request.url.path.startswith("/api") or "application/json" not in content_type:
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])
    try: payload = json.loads(body or b"null")
    except json.JSONDecodeError: return Response(body, status_code=response.status_code, headers=dict(response.headers), media_type=content_type)
    if isinstance(payload, dict) and "success" in payload: wrapped = payload
    elif response.status_code >= 400:
        message = payload.get("detail", "Request failed") if isinstance(payload, dict) else "Request failed"
        wrapped = {"success": False, "message": message if isinstance(message, str) else "Request validation failed", "details": payload}
    else:
        wrapped = {"success": True, "message": "Request completed successfully.", "data": payload}
    headers = dict(response.headers); headers.pop("content-length", None); headers.pop("content-type", None)
    return Response(json.dumps(wrapped, ensure_ascii=True, allow_nan=False), response.status_code, headers, media_type="application/json")
