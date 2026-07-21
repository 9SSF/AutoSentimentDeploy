基于 FastAPI 和 Hugging Face Transformers 的情感分析 API 服务，下载模型到本地并且部署，支持 Docker 容器化与 CI/CD 自动化流水线


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python3 -m app.main
python3 -m pytest
python3 -m pytest -s # 打印输出

deactivate # exit
```

docker:
```bash
docker build -t auto-sentiment:v1 . #构建镜像
docker run -d -p 8000:8000 --name auto-sentiment auto-sentiment:v1 # 启动容器

docker logs -f auto-sentiment # 验证日志
```