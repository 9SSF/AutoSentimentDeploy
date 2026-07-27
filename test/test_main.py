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
        text_lower = text.lower()  
        is_positive = "love" in text_lower or "great" in text_lower
        return {
            "label": "POSITIVE" if is_positive else "NEGATIVE",
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
def test_health_check(client):
    """测试健康检查接口 /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_name" in data

def test_predict_single_success(client):
    """测试正常输入时, API 能否正确返回 200 和预期的结构"""
    response = client.post("/predict", json={"text": "I love this project!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["model"] == "test-fake-model"

def test_predict_single_blank_text(client):
    """测试输入空格时, Pydantic 数据校验是否能成功拦截并返回 422 报错"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_batch_success(client):
    """测试批量预测接口 /predict/batch 成功分支"""
    payload = {"texts": ["Great product!", "Terrible experience."]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["label"] == "POSITIVE"
    assert data["results"][1]["label"] == "NEGATIVE"

def test_predict_batch_too_many_items(client):
    """测试批量预测数量超过限制 (超出 settings.max_batch_size)"""
    # 构造超过限制的大数组
    payload = {"texts": ["test"] * 100}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """测试 /metrics 监控端点是否能正常返回 Prometheus 数据"""
    response = client.get("/metrics")
    assert response.status_code == 200
    # 验证返回内容里是否包含了我们定义的指标名称
    assert "http_requests_total" in response.text
    assert "predictions_total" in response.text

def test_metrics_increment_on_predict(client):
    """测试调用预测接口后，Prometheus 指标是否包含了 POSITIVE 标签统计"""
    # 1. 先请求一次预测
    response = client.post("/predict", json={"text": "I love FastAPI!"})
    assert response.status_code == 200
    assert response.json()["label"] == "POSITIVE"

    # 2. 再次请求 /metrics
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    
    # 3. 灵活验证：断言指标文本里同时包含指标名、POSITIVE 标签和 fake 模型名
    metrics_text = metrics_response.text
    assert "predictions_total" in metrics_text
    assert 'label="POSITIVE"' in metrics_text
    assert 'model="test-fake-model"' in metrics_text