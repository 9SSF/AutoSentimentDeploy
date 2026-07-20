import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 文件中的变量到系统环境变量中
load_dotenv()

@dataclass(frozen=True)
class Settings:
    # 定义配置字段，并设置默认值（如果环境变量里没有，就用默认值）
    model_name: str = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "512"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

# 实例化配置对象，供其他文件导入
settings = Settings()