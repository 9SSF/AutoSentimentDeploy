from transformers import pipeline


print("正在加载模型...")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print("模型加载成功！")


def get_prediction(text: str):
    # 直接调用全局的模型变量进行预测
    raw_result = classifier(text)
    return raw_result[0]