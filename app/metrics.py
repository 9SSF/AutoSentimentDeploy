# app/metrics.py
from prometheus_client import Counter, Histogram

# 1. 计数器：统计所有 HTTP 请求的总次数（按请求方法、接口路由、状态码分类）
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "endpoint", "status_code"],
)

# 2. 直方图：统计 HTTP 请求的耗时分布（单位：秒）
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求处理耗时（秒）",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# 3. 业务计数器：统计 AI 模型预测出的不同情感标签（POSITIVE / NEGATIVE）的数量
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "模型预测情感标签总数",
    ["label", "model"],
)