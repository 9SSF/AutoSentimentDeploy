import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model import model_service

# 1. 创建一个假的 AI 模型服务来替代真实的模型服务
class FakeSentimentService:
    model_name = "test-fake-model"
    loaded = True

    def load(self) -> None:
        self.loaded = True

    def predict(self, text: str):
        # 模拟真实的推理返回结构
        print("\n>>> 正在使用【假模型】进行预测！ <<<")  
        return {
            "label": "POSITIVE" if "love" in text.lower() else "NEGATIVE",
            "score": 0.99
        }

# 2. Pytest 的 Fixture，在测试开始前把真实模型替换成假模型
@pytest.fixture
def client(monkeypatch): # 使用 pytest 自带的 monkeypatch 补丁工具
    fake = FakeSentimentService()
    # 把 model_service 替换掉
    monkeypatch.setattr("app.main.model_service", fake)
    
    with TestClient(app) as test_client:
        yield test_client


# 3. 编写测试用例
def test_predict_success(client):
    """测试正常输入时, API 能否正确返回 200 和预期的结构"""
    response = client.post("/predict", json={"text": "I love this project!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["model"] == "test-fake-model"

def test_predict_blank_text(client):
    """测试输入空格时, Pydantic 数据校验是否能成功拦截并返回 422 报错"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422