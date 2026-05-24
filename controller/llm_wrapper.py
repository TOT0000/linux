import os
import openai
from openai import Stream, ChatCompletion

from .utils import print_debug

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o")
GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")
LMSTUDIO_DEFAULT_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "local-model")


def _normalize_provider(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if value in {"openai", "gemini", "lmstudio"}:
        return value
    return ""


def _resolve_default_provider() -> str:
    forced = _normalize_provider(os.environ.get("LLM_PROVIDER"))
    if forced:
        return forced
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "gemini"


def _provider_default_model(provider: str) -> str:
    if provider == "openai":
        return OPENAI_DEFAULT_MODEL
    if provider == "lmstudio":
        return LMSTUDIO_DEFAULT_MODEL
    return GEMINI_DEFAULT_MODEL


DEFAULT_PROVIDER = _resolve_default_provider()
GEMINI_MODEL = _provider_default_model(DEFAULT_PROVIDER)
MODEL_NAME = GEMINI_MODEL

# Keep legacy aliases for compatibility with existing callers/UI toggles.
GPT3 = MODEL_NAME
GPT4 = MODEL_NAME
LLAMA3 = MODEL_NAME

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

def _mask_key(secret: str | None) -> str:
    value = "" if secret is None else str(secret).strip()
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return f"{value[:2]}***{value[-2:]}"
    return f"{value[:6]}...{value[-4:]}"

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.provider = _resolve_default_provider()
        self.base_url = (
            OPENAI_BASE_URL
            if self.provider == "openai"
            else (LMSTUDIO_BASE_URL if self.provider == "lmstudio" else GEMINI_BASE_URL)
        )
        if self.provider == "openai":
            self.api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
            self.key_source = ("OPENAI_API_KEY" if self.api_key else "(empty)")
        elif self.provider == "lmstudio":
            self.api_key = (os.environ.get("LMSTUDIO_API_KEY") or "lmstudio").strip()
            self.key_source = "LMSTUDIO_API_KEY"
        else:
            self.api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            ).strip()
            if os.environ.get("GEMINI_API_KEY"):
                self.key_source = "GEMINI_API_KEY"
            elif os.environ.get("GOOGLE_API_KEY"):
                self.key_source = "GOOGLE_API_KEY"
            else:
                self.key_source = "(empty)"
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        print_debug(
            "[LLM-CONFIG] "
            f"provider={self.provider} "
            f"has_gemini_key={bool(os.environ.get('GEMINI_API_KEY'))} "
            f"has_openai_key={bool(os.environ.get('OPENAI_API_KEY'))} "
            f"has_google_key={bool(os.environ.get('GOOGLE_API_KEY'))} "
            f"selected_key_len={len(self.api_key)} "
            f"selected_key_masked={_mask_key(self.api_key)} "
            f"key_source={self.key_source} "
            f"base_url={self.base_url} "
            f"default_model={_provider_default_model(self.provider)}"
        )

    def request(self, prompt, model_name=GEMINI_MODEL, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        selected_model = str(model_name or _provider_default_model(self.provider))
        # Provider/model safety guard to avoid mismatches.
        if self.provider == "gemini" and selected_model.lower().startswith("gpt-"):
            selected_model = GEMINI_DEFAULT_MODEL
        elif self.provider == "openai" and selected_model.lower().startswith("gemini-"):
            selected_model = OPENAI_DEFAULT_MODEL
        print_debug(
            f"[LLM] provider={self.provider}"
        )
        print_debug(
            f"[LLM] base_url={self.base_url}"
        )
        print_debug(
            f"[LLM] model_name={selected_model}"
        )
        print_debug(
            f"[LLM] key_source={self.key_source}"
        )

        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
        print_debug(f"[LLM] Prompt written to {chat_log_path}")
        
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=stream,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response

        return response.choices[0].message.content
