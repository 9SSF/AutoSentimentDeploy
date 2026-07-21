from transformers import pipeline

from app.settings import settings

class SentimentModelService:
    def __init__(self):
        # 初始时不加载模型，省内存
        self._classifier = None  
        self.model_name = settings.model_name

    @property
    def loaded(self) -> bool:
        return self._classifier is not None

    def load(self) -> None:
        """这个方法只在服务器真正启动时由寿命周期管理器调用一次"""
        if self._classifier is None:
            print(">>> 【Service】正在从缓存/网络加载大模型...")
            self._classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            print(">>> 【Service】大模型加载成功！已成功驻留内存！")

    def predict(self, text: str):
        if not self.loaded:
            return {"error": "模型尚未加载就绪"}
        # 实际调用推理
        raw_result = self._classifier(text)
        return raw_result[0]

# 实例化
model_service = SentimentModelService()