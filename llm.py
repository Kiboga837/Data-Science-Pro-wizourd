"""AI provider layer for Questions & Answers and the report summary.

Default: Google Gemini on the FREE tier (no billing or credit card; key from Google AI Studio).
Optional: Anthropic Claude (paid). Both expose the same `generate()` method.

Gemini is called through its plain REST API with the standard library, so no extra package is needed.
The free tier has request-per-minute and per-day limits that Google does not publish in its docs (they are shown in
the AI Studio rate-limit page), so this client paces requests and retries politely.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import streamlit as st

PROVIDER_LABELS = ["Google Gemini (free tier)", "Anthropic Claude (paid)"]
PROVIDERS = {PROVIDER_LABELS[0]: "gemini", PROVIDER_LABELS[1]: "anthropic"}
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-5-5"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
KEY_HELP = {"gemini": "Free key, no credit card: https://aistudio.google.com/app/apikey",
            "anthropic": "Paid key: https://console.anthropic.com"}
_last_call = {}  # per-key timestamp of the last request, used for pacing


class LLMError(Exception):
    """kind: auth | quota | rate | model | network | blocked | other"""

    def __init__(self, message: str, kind: str = "other"):
        super().__init__(message)
        self.kind = kind


@dataclass
class Credentials:
    provider: str
    key: str
    model: str
    pause: float = 4.0
    economy: bool = False


# ----------------------------------------------------------------------------
# HTTP (a single function so tests can replace it)
# ----------------------------------------------------------------------------
def _http(method: str, url: str, headers: dict, body: dict = None, timeout: float = 90.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        return e.code, payload
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Could not reach the AI service ({e}). Check your internet connection.", "network") from e


def _error_message(payload: dict) -> str:
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    return str(err.get("message", "")) or "no details"


def _retry_delay(payload: dict, default: float) -> float:
    for d in (payload.get("error", {}) or {}).get("details", []) or []:
        val = d.get("retryDelay")
        if val:
            try:
                return min(float(str(val).rstrip("s")), 90.0)
            except ValueError:
                pass
    return default


# ----------------------------------------------------------------------------
# Clients
# ----------------------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str, pause: float = 4.0, max_retries: int = 4, notify=None):
        self.key, self.pause, self.max_retries, self.notify = api_key, pause, max_retries, notify
        self.use_thinking_off = True

    def _headers(self):
        return {"Content-Type": "application/json", "x-goog-api-key": self.key}

    def _pace(self):
        wait = self.pause - (time.time() - _last_call.get(self.key, 0))
        if wait > 0:
            time.sleep(wait)
        _last_call[self.key] = time.time()

    def generate(self, model: str, system: str, messages: list, max_tokens: int = 3000) -> str:
        model = model.replace("models/", "")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages]
        # thinking tokens share the output budget on Gemini 2.5, so leave room and switch thinking off when allowed
        config = {"maxOutputTokens": max(max_tokens, 4096), "temperature": 0.2}
        delays = [8.0, 20.0, 40.0, 60.0]
        for attempt in range(self.max_retries + 1):
            body = {"contents": contents, "generationConfig": dict(config)}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if self.use_thinking_off and "flash" in model:
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
            self._pace()
            status, payload = _http("POST", f"{GEMINI_URL}/models/{model}:generateContent", self._headers(), body)
            if status == 200:
                return self._extract(payload)
            msg = _error_message(payload)
            low = msg.lower()
            if status == 400 and "thinking" in low and self.use_thinking_off:
                self.use_thinking_off = False  # this model does not accept the setting; retry without it
                continue
            if status == 400 and ("api key" in low or "api_key" in low):
                raise LLMError("Gemini rejected the API key. Create a free key at "
                               "https://aistudio.google.com/app/apikey and paste it again.", "auth")
            if status in (401, 403):
                raise LLMError(f"Gemini refused the request ({msg}). Check the key, and that the Generative "
                               "Language API is allowed in your region and project.", "auth")
            if status == 404:
                raise LLMError(f"Model '{model}' was not found. Press 'List available models' and pick one.", "model")
            if status == 429:
                if "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low:
                    raise LLMError("The free daily request limit for this model is used up. It resets daily; "
                                   "try another model (for example gemini-2.5-flash-lite) or come back later.", "quota")
                if attempt < self.max_retries:
                    wait = _retry_delay(payload, delays[min(attempt, len(delays) - 1)])
                    if self.notify:
                        self.notify(f"Free-tier rate limit reached; waiting {wait:.0f}s then retrying...")
                    time.sleep(wait)
                    continue
                raise LLMError("The free-tier rate limit is still being hit. Wait a minute, or raise the pause "
                               "between requests, then press the button again (finished answers are kept).", "rate")
            if status >= 500 and attempt < self.max_retries:
                time.sleep(delays[min(attempt, len(delays) - 1)] / 2)
                continue
            raise LLMError(f"Gemini error {status}: {msg}", "other")
        raise LLMError("Gemini did not return an answer.", "other")

    @staticmethod
    def _extract(payload: dict) -> str:
        cands = payload.get("candidates") or []
        if not cands:
            reason = (payload.get("promptFeedback") or {}).get("blockReason")
            raise LLMError("Gemini returned no answer" + (f" (blocked: {reason})." if reason else "."), "blocked")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"Gemini returned an empty answer (finish reason: {cands[0].get('finishReason')}).", "blocked")
        return text

    def list_models(self) -> list:
        status, payload = _http("GET", f"{GEMINI_URL}/models?pageSize=100", self._headers())
        if status != 200:
            raise LLMError(f"Could not list models: {_error_message(payload)}", "auth" if status in (400, 401, 403) else "other")
        return sorted(m["name"].replace("models/", "") for m in payload.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []) and "gemini" in m["name"])


class AnthropicClient:
    def __init__(self, api_key: str, **_):
        self.key = api_key

    def generate(self, model: str, system: str, messages: list, max_tokens: int = 3000) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("The `anthropic` package is not installed (pip install anthropic).", "other") from e
        try:
            client = anthropic.Anthropic(api_key=self.key)
            resp = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        except Exception as e:  # noqa: BLE001
            kind = "auth" if "Authentication" in type(e).__name__ else "other"
            raise LLMError(f"Anthropic error: {e}", kind) from e

    def list_models(self) -> list:
        return []


def make_client(creds: Credentials, notify=None):
    if creds.provider == "gemini":
        return GeminiClient(creds.key, pause=creds.pause, notify=notify)
    return AnthropicClient(creds.key)


# ----------------------------------------------------------------------------
# Credentials (secrets > environment > what the user typed this session)
# ----------------------------------------------------------------------------
def _secret_key(provider: str) -> str:
    names = ("GEMINI_API_KEY", "GOOGLE_API_KEY") if provider == "gemini" else ("ANTHROPIC_API_KEY",)
    for name in names:
        try:
            val = st.secrets.get(name, "")
        except Exception:  # noqa: BLE001
            val = ""
        val = val or os.environ.get(name, "")
        if val:
            return str(val)
    return ""


def get_credentials() -> Credentials:
    ss = st.session_state
    provider = PROVIDERS.get(ss.get("api_provider", PROVIDER_LABELS[0]), "gemini")
    key = _secret_key(provider) or str(ss.get(f"api_key_{provider}", ""))
    model = str(ss.get(f"api_model_{provider}") or DEFAULT_MODELS[provider])
    try:
        pause = float(ss.get("api_pause", 4.0))
    except (TypeError, ValueError):
        pause = 4.0
    return Credentials(provider, key.strip(), model.strip(), pause, bool(ss.get("api_economy", False)))


def render_llm_settings() -> Credentials:
    """Provider, key, model and quota settings. Returns the resulting credentials."""
    st.session_state.setdefault("api_provider", PROVIDER_LABELS[0])
    st.selectbox("AI provider", PROVIDER_LABELS, key="api_provider")
    provider = PROVIDERS[st.session_state["api_provider"]]
    st.session_state.setdefault(f"api_model_{provider}", DEFAULT_MODELS[provider])
    st.session_state.setdefault("api_pause", 4.0)
    st.session_state.setdefault("api_economy", False)

    a, b = st.columns(2)
    secret = _secret_key(provider)
    if secret:
        a.success("API key loaded from secrets or environment.")
    else:
        a.text_input("API key", type="password", key=f"api_key_{provider}",
                     help=KEY_HELP[provider] + ". Kept only for this session.")
        a.caption(KEY_HELP[provider])
    b.text_input("Model", key=f"api_model_{provider}")

    if provider == "gemini":
        c, d = st.columns(2)
        c.number_input("Pause between requests (seconds)", 0.0, 60.0, step=1.0, key="api_pause",
                       help="The free tier allows only a few requests per minute. A longer pause avoids rate-limit waits.")
        d.checkbox("Save quota: skip the written explanation (each question then uses 1 request instead of 2)",
                   key="api_economy")
        st.caption("Free tier needs no billing. Google's pricing page says free-tier requests may be used to improve "
                   "its products, so avoid sending confidential data. Per-model limits are shown in Google AI "
                   "Studio under 'Rate limits'.")
    creds = get_credentials()

    t1, t2 = st.columns(2)
    if t1.button("Test connection"):
        if not creds.key:
            st.error("Enter an API key first.")
        else:
            try:
                reply = make_client(creds).generate(creds.model, "", [{"role": "user", "content": "Reply with the single word OK."}], 50)
                st.success(f"Connected to {creds.model}. Reply: {reply.strip()[:40]}")
            except LLMError as e:
                st.error(str(e))
    if provider == "gemini" and t2.button("List available models"):
        if not creds.key:
            st.error("Enter an API key first.")
        else:
            try:
                names = make_client(creds).list_models()
                st.info("Models that support text generation for your key: " + ", ".join(names))
                st.caption("Free-tier models are usually the 'flash' and 'flash-lite' ones; preview models may need billing.")
            except LLMError as e:
                st.error(str(e))
    return creds
