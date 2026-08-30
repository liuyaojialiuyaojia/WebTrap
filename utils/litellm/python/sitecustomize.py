# 自动补丁：把 openai.OpenAI 的 base_url / api_key 强制指向代理
import os
import sys
import importlib.abc
from importlib.machinery import PathFinder


def _patch_openai(mod):
    try:
        OpenAI = mod.OpenAI
    except Exception:
        return

    orig_init = OpenAI.__init__

    def patched_init(self, *args, **kwargs):
        if os.getenv("LITELLM_BYPASS") == "1":
            return orig_init(self, *args, **kwargs)
        base_url = os.getenv("LITELLM_BASE_URL")
        api_key = os.getenv("LITELLM_MASTER_KEY") or os.getenv("OPENAI_API_KEY")
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return orig_init(self, *args, **kwargs)

    OpenAI.__init__ = patched_init


try:
    import openai as _openai

    _patch_openai(_openai)
except Exception:

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "openai":
                return None
            spec = PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            orig_loader = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, s):
                    if hasattr(orig_loader, "create_module"):
                        return orig_loader.create_module(s)

                def exec_module(self, module):
                    orig_loader.exec_module(module)
                    _patch_openai(module)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Finder())
