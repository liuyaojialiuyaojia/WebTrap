import os

from utils.litellm.python.client import chat

os.environ.setdefault("LITELLM_BASE_URL", "http://localhost:4000/v1")
os.environ.setdefault("LITELLM_MASTER_KEY", "sk-yaojia-get-ccfa")
SMOKE_MODEL = os.getenv("SMOKE_MODEL", "gpt-4o")

response = chat(
    messages=[{"role": "user", "content": "用一句中文自我介绍，十个字以内"}],
    model=SMOKE_MODEL,
    cache_ttl=300,
    namespace="smoke",
)
print(response.choices[0].message.content)
