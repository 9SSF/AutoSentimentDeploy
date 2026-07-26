"""
把日志转为单行的json格式，方便日志收集和分析
"""

# app/logging_config.py
import logging
import json
import time
from contextvars import ContextVar

# 使用 ContextVar 存储当前协程/线程上下文中的 request_id，实现跨函数全局追踪
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

class JSONFormatter(logging.Formatter):
    """自定义 JSON 格式化器，把 log 记录转换为标准 JSON"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        
        # 如果日志包含异常堆栈，一并塞入 JSON
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging(log_level: str = "INFO"):
    """全局初始化日志配置"""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除默认的 handler 防止日志重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)