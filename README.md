基于 FastAPI 和 Hugging Face Transformers 的情感分析 API 服务，下载模型到本地并且部署，支持 Docker 容器化与 CI/CD 自动化流水线


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python3 main.py

deactivate # exit
```