from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
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

# 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 当 Uvicorn 真正启动 server 进程时，这里才会被触发【仅一次】
    model_service.load()
    yield
    print(">>> 【Lifespan】服务器正在关闭...")

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

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest):
    if not model_service.loaded:
        return {"error": "模型尚未准备就绪"}
    # 对文本长度进行限制
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(
            status_code=422, 
            detail=f"文本长度不能超过 {settings.max_text_length} 个字符"
        )
    # payload.text 让 IDE 能自动补全
    raw_result = model_service.predict(payload.text)
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