import time

from starlette.middleware.base import BaseHTTPMiddleware

from src.api.core.logging import get_logger

logger = get_logger(__name__)

class LatencyMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000  # ms

        logger.info(
            f"{request.method} {request.url.path} - {process_time:.2f}ms"
        )

        return response
