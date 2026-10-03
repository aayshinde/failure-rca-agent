"""Chat model factory + a callback that counts tokens and dollars per request."""
from __future__ import annotations

import threading
from functools import lru_cache

from langchain_core.callbacks import BaseCallbackHandler

from src import config


@lru_cache(maxsize=4)
def get_chat(model: str | None = None, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    if config.LLM_PROVIDER == "ollama":   # Ollama speaks the OpenAI protocol; local models are slower, so allow more time
        return ChatOpenAI(model=model or config.CHAT_MODEL, temperature=temperature, base_url=config.OLLAMA_URL,
                          api_key="ollama", max_retries=2, timeout=300)
    return ChatOpenAI(model=model or config.CHAT_MODEL, temperature=temperature, max_retries=6, timeout=90)


def structured(llm, schema):
    """Structured-output chain. OpenAI models use function calling; local models are far more reliable with
    Ollama's JSON-schema constrained decoding. Fakes used in tests ignore the method argument."""
    method = "json_schema" if config.LLM_PROVIDER == "ollama" else "function_calling"
    return llm.with_structured_output(schema, method=method)


class UsageTracker(BaseCallbackHandler):
    """Pass as config={"callbacks": [tracker]}; works through LangGraph and structured output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                model = (getattr(msg, "response_metadata", None) or {}).get("model_name", config.CHAT_MODEL)
                i, o = um.get("input_tokens", 0), um.get("output_tokens", 0)
                pin, pout = next((v for k, v in config.PRICES.items() if str(model).startswith(k)), (0.0, 0.0))
                with self.lock:
                    self.calls += 1
                    self.input_tokens += i
                    self.output_tokens += o
                    self.cost_usd += (i * pin + o * pout) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6)}
