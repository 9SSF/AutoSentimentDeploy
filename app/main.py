import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
import uvicorn

from app.model import model_service
from app.schemas import (
    PredictRequest, 
    PredictionResponse, 
    BatchPredictRequest, 
    BatchPredictionResponse, 
    HealthResponse
)
from app.settings import settings
from app.logging_config import setup_logging, request_id_ctx

# 1. 初始化结构化日志
setup_logging(settings.log_level)
logger = logging.getLogger("app.main")

# 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> 【Lifespan】正在加载模型...")
    # 当 Uvicorn 真正启动 server 进程时，这里才会被触发【仅一次】
    model_service.load()
    logger.info(">>> 【Lifespan】模型加载完成，服务就绪！")
    yield
    logger.info(">>> 【Lifespan】服务器正在关闭...")

# 将寿命周期管理器注册进 FastAPI
app = FastAPI(lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health():
    """健康检查"""
    return HealthResponse(
        status="ok" if model_service.loaded else "degraded",
        model_loaded=model_service.loaded,
        model_name=settings.model_name,
    )

# 2. 全局 HTTP 中间件：拦截请求、记录耗时、追踪 request_id
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    
    # 优先从 Request Header 中获取调用方传来的 x-request-id，没有则自动生成 UUID
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_ctx.set(req_id)
    
    try:
        response: Response = await call_next(request)
        process_time = (time.time() - start_time) * 1000  # 算出的毫秒数
        
        # 记录结构化请求日志
        logger.info(
            f"HTTP {request.method} {request.url.path} - Status: {response.status_code} - Latency: {process_time:.2f}ms"
        )
        # 将 request_id 返回给前端/客户端，方便排查
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        request_id_ctx.reset(token)

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest):
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="模型尚未准备就绪")
    # 对文本长度进行限制
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(
            status_code=422, 
            detail=f"文本长度不能超过 {settings.max_text_length} 个字符"
        )
    # payload.text 让 IDE 能自动补全
    raw_result = model_service.predict(payload.text)
    logger.info(f"predict_completed label={raw_result['label']} score={raw_result['score']:.4f}")
    # 显式组装符合 PredictionResponse 要求的字典
    return {
        "label": str(raw_result["label"]),
        "score": float(raw_result["score"]),
        "model": model_service.model_name
    }
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictRequest):
    """批量预测接口"""
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="模型尚未就绪")
    if len(payload.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=422, 
            detail=f"批量处理数量不能超过 {settings.max_batch_size}"
        )
    if any(len(text) > settings.max_text_length for text in payload.texts):
        raise HTTPException(
            status_code=422, 
            detail=f"每条文本长度不能超过 {settings.max_text_length} 个字符"
        )
    # 对每个文本进行预测
    results = []
    for text in payload.texts:
        raw_result = model_service.predict(text)
        results.append({
            "label": str(raw_result["label"]),
            "score": float(raw_result["score"]),
            "model": model_service.model_name
        })
    return BatchPredictionResponse(results=results, model=model_service.model_name)


if __name__ == "__main__":
    # 启动一个运行在 8000 端口的 Web 服务器，并且检测到代码修改时自动重启(--reload)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)