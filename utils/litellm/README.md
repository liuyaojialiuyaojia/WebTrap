# 项目说明

## 启动
- 复制 `.env.example` 为 `.env` 并填好密钥。
- 运行 `utils/litellm/scripts/up.sh` 启动代理与 Redis。
- 运行 `utils/litellm/scripts/smoke.sh` 做一次冒烟调用。
- 若宿主机需要走代理才能访问上游（或直连会触发区域限制），可在 `utils/litellm/.env` 里补充：
  - `HTTP_PROXY=http://host.docker.internal:<port>`
  - `HTTPS_PROXY=http://host.docker.internal:<port>`
  - `NO_PROXY=localhost,127.0.0.1,redis,host.docker.internal`
  然后运行 `utils/litellm/scripts/reload.sh` 让容器生效。
- 当前 Compose 中的 `proxy-bridge` 会把 Docker 网桥上的
  `172.17.0.1:17891` 转发到宿主机回环地址 `127.0.0.1:17891`，用于访问
  仅监听本机的 Clash；它只监听 Docker 网桥，不会把代理端口暴露到局域网。
- 默认 Redis 仅在 Docker 网络内暴露（无宿主机端口冲突）。若需从宿主机直接访问，可在 `docker-compose.yml` 为 `redis` 服务手动补充 `ports` 映射。
- `up.sh` 会自动兼容 `docker compose` 和 `docker-compose`，并以 Compose 项目名 `litellm` 启动三个容器：
  - `litellm-litellm-1`：LiteLLM 代理，监听 4000，对外暴露 OpenAI 兼容接口。
  - `litellm-redis-1`：Redis 缓存，只在 Compose 网络内开放。
  - `litellm-proxy-bridge-1`：把容器请求安全转发到宿主机回环地址上的代理。
  可通过 `docker ps --filter label=com.docker.compose.project=litellm` 或
  `docker-compose -f utils/litellm/docker-compose.yml ps` 随时核对它们是否仍在运行；
  `utils/litellm/scripts/down.sh` 会一次停掉这三个容器。

## 零侵入运行现有程序
- 用 `utils/litellm/scripts/run.sh <你的启动命令>` 启动现有程序。
- run.sh 会自动把 OpenAI SDK 指向本代理并启用缓存。

## 在新项目中直接使用
- `from utils.litellm.python.client import chat`，保持 OpenAI 兼容请求体。
- 在每次调用里用 `cache_ttl` 或 `cache_options` 控制缓存策略。

## 运维
- `j` 重载代理配置。
- `utils/litellm/scripts/down.sh` 停止服务。

# 附录：LiteLLM 缓存访问速查

**命令行**
- 请求时附带 `cache` 字段即可写入缓存：
  ```bash
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy localhost \
      http://localhost:4000/v1/chat/completions \
      -H 'Authorization: Bearer sk-your-litellm-key' \
      -H 'Content-Type: application/json' \
      -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"Give me one fun trivia fact."}],"cache":{"ttl":600,"namespace":"demo"}}'
  ```
- 查看缓存命中可加 `-i` 观察 `x-litellm-cache-key` / `x-litellm-cache-hit`，或列出指定命名空间：
  ```bash
  curl --noproxy localhost \
    'http://localhost:4000/cache/list?namespace=demo' \
    -H 'Authorization: Bearer sk-your-litellm-key'
  ```

**代码示例**
- 利用仓库自带封装（`utils/litellm/python/client.py`）附带缓存参数：
  ```python
  from utils.litellm.python.client import chat

  response = chat(
      model="openai/gpt-4o-mini",
      messages=[{"role": "user", "content": "Give me one fun trivia fact."}],
      cache_ttl=600,
      namespace="demo",
  )

  print(response.choices[0].message.content)
  ```
- 若需禁用缓存，可传 `cache_ttl=None` 并显式设置 `cache_options={"no-cache": True}`。
