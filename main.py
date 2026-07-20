from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from model import model_service
from schemas import PredictRequest, PredictionResponse


# 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 当 Uvicorn 真正启动 server 进程时，这里才会被触发【仅一次】
    model_service.load()
    yield
    print(">>> 【Lifespan】服务器正在关闭...")

# 将寿命周期管理器注册进 FastAPI[cite: 1]
app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest):
    if not model_service.loaded:
        return {"error": "模型尚未准备就绪"}
    
    # payload.text 让 IDE 能自动补全，再也不会打错字
    raw_result = model_service.predict(payload.text)
    
    # 显式组装符合 PredictionResponse 要求的字典[cite: 5]
    return {
        "label": str(raw_result["label"]),
        "score": float(raw_result["score"]),
        "model": "distilbert-base-uncased-finetuned-sst-2-english"
    }

if __name__ == "__main__":
    # 启动一个运行在 8000 端口的 Web 服务器，并且检测到代码修改时自动重启(--reload)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)