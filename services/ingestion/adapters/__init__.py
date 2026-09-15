from adapters.base import BaseAdapter
from adapters.local_json import LocalJsonAdapter
from adapters.epub import EpubAdapter


def get_adapter(source_type: str, source_path: str) -> BaseAdapter:
    adapters = {
        "local_json": LocalJsonAdapter,
        "epub": EpubAdapter,
    }
    adapter_cls = adapters.get(source_type)
    if not adapter_cls:
        raise ValueError(f"Unknown source type: {source_type}. Available: {list(adapters.keys())}")
    return adapter_cls(source_path)
