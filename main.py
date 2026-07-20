from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()

print("正在加载模型...")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print("模型加载成功！")

# 新手做法：直接接收一个普通的 Python 字典（Dict），不做任何严格的格式校验
@app.post("/predict")
def predict(data: dict):
    # 手动判断用户有没有传 text 过来
    if "text" not in data or not data["text"]:
        return {"error": "请输入有效的文本"}
    
    text = data["text"]
    
    # 直接调用全局的模型变量进行预测
    raw_result = classifier(text)
    return raw_result[0]

if __name__ == "__main__":
    # 启动一个运行在 8000 端口的 Web 服务器，并且检测到代码修改时自动重启(--reload)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)