from pydantic import BaseModel, Field, field_validator

# 定义输入格式，限制文本不能为空
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

# 定义输出格式
class PredictionResponse(BaseModel):
    label: str
    score: float
    model: str

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    model: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str