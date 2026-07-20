from pydantic import BaseModel, Field

# 定义输入格式，限制文本不能为空
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

# 定义输出格式
class PredictionResponse(BaseModel):
    label: str
    score: float
    model: str