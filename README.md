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

docker compose up --build # 进入交互式 TUI 模式
```

新开端口请求：
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"FastAPI with Prometheus is awesome!"}'
```

打开prometheus：
访问 http://localhost:9090。在搜索框输入 http_requests_total 并点击 Execute，看到实时更新的指标

打开 Grafana：
访问 http://localhost:3000。 默认账号：admin  默认密码：admin  进入后点击左侧 Explore，数据源选择 Prometheus，右上角选择 code，输入查询语句 `sum(rate(http_requests_total[1m]))`，就能实时看到 API QPS 请求折线图了！


```bash
# 启动docker后查看docker的占用空间
docker builder prune -a

# 清理无用的构建缓存
docker builder prune -a

# 清理未使用的旧镜像和停止的容器
docker image prune -a

# 清理已停止的容器、未被使用的网络、悬空镜像、悬空的构建缓存
docker system prune
```

```bash
# 查看挂载卷的位置
docker volume inspect huggingface-cache
```