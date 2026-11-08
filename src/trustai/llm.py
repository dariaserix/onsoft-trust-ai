"""Подключение к языковой модели.

Провайдеры (выбираются в файле .env, код менять не нужно):
- gigachat : GigaChat API от Сбера (работает в РФ без VPN);
- openai   : любой OpenAI-совместимый API (YandexGPT, локальная Ollama и т.п.).

Каждый запрос пишется в журнал results/journal.jsonl (без ключей): время, модель, токены, длительность.
Ответы кэшируются в results/cache: повторный прогон того же запроса бесплатен и даёт тот же результат.
"""
import base64
import re
import hashlib
import json
import os
import ssl
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path=None):
    """Читает .env (KEY=VALUE) без сторонних библиотек. Переменные окружения важнее файла."""
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class LLMResult:
    text: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLMError(RuntimeError):
    pass


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Модели с рассуждениями (Qwen3 и др.) пишут ход мысли в <think>...</think>. Нам нужен только ответ."""
    text = text or ""
    if "</think>" in text.lower():
        text = _THINK_RE.sub("", text)
        low = text.lower()
        if "</think>" in low:  # открывающего тега не было
            text = text[low.rfind("</think>") + len("</think>"):]
    return text.strip()


class BaseLLM:
    provider = "base"

    def __init__(self, model, temperature=0.1, max_tokens=1200, cache_dir=None, journal=None, timeout=120,
                 use_cache=True):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir or ROOT / "results" / "cache")
        self.journal = Path(journal or ROOT / "results" / "journal.jsonl")
        self.timeout = timeout
        self.use_cache = use_cache
        self.stats = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    @property
    def name(self):
        return f"{self.provider}:{self.model}"

    def _key(self, messages):
        blob = json.dumps([self.provider, self.model, self.temperature, self.max_tokens, messages],
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages, purpose="", qid="") -> LLMResult:
        key = self._key(messages)
        cpath = self.cache_dir / key[:2] / f"{key}.json"
        if self.use_cache and cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            res = LLMResult(data["text"], True, data.get("prompt_tokens", 0), data.get("completion_tokens", 0), 0.0)
        else:
            t0 = time.time()
            text, pt, ct = self._call(messages)
            res = LLMResult(text, False, pt, ct, time.time() - t0)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps({"text": text, "prompt_tokens": pt, "completion_tokens": ct,
                                         "model": self.name}, ensure_ascii=False), encoding="utf-8")
        self.stats["calls"] += 1
        self.stats["cached"] += int(res.cached)
        self.stats["prompt_tokens"] += res.prompt_tokens
        self.stats["completion_tokens"] += res.completion_tokens
        self.stats["seconds"] += res.seconds
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "qid": qid, "purpose": purpose,
                                "model": self.name, "cached": res.cached, "prompt_tokens": res.prompt_tokens,
                                "completion_tokens": res.completion_tokens, "seconds": round(res.seconds, 2)},
                               ensure_ascii=False) + "\n")
        return res

    def _call(self, messages):
        raise NotImplementedError

    # --- общий HTTP с повторами ---
    def _post(self, url, headers, body: bytes, ctx=None):
        delay = 2.0
        last = None
        for attempt in range(6):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last = LLMError(f"HTTP {e.code}: {detail}")
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(delay)
                    delay *= 2
                    continue
                if e.code == 401 and hasattr(self, "_token"):
                    self._token = None  # токен истёк: получим новый
                    headers = self._headers()
                    continue
                raise last
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = LLMError(f"Сеть: {e}")
                time.sleep(delay)
                delay *= 2
        raise last


class GigaChat(BaseLLM):
    provider = "gigachat"
    OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    def __init__(self, model=None, **kw):
        super().__init__(model or os.environ.get("GIGACHAT_MODEL", "GigaChat-2-Max"), **kw)
        self.auth_key = os.environ.get("GIGACHAT_AUTH_KEY", "").strip()
        self.scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self._token, self._exp = None, 0
        self._ctx = self._ssl()

    CERT_FILE = ROOT / "certs" / "russian_trusted_root_ca.cer"
    CERT_URLS = ["https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt",
                 "https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer"]

    def _auto_cert(self):
        """Скачивает корневой сертификат Минцифры (нужен для GigaChat) в certs/."""
        for url in self.CERT_URLS:
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                if data:
                    self.CERT_FILE.parent.mkdir(parents=True, exist_ok=True)
                    self.CERT_FILE.write_bytes(data)
                    print(f"  Скачан сертификат Минцифры: {self.CERT_FILE}")
                    return str(self.CERT_FILE)
            except Exception as e:
                print(f"  Не удалось скачать сертификат с {url}: {e}")
        return None

    def _ssl(self):
        bundle = os.environ.get("GIGACHAT_CA_BUNDLE", "").strip()
        if not bundle and self.CERT_FILE.exists():
            bundle = str(self.CERT_FILE)
        if bundle:
            p = Path(bundle)
            if not p.is_absolute():
                p = ROOT / p
            ctx = ssl.create_default_context()  # системные сертификаты + сертификат Минцифры
            data = p.read_bytes()
            if b"BEGIN CERTIFICATE" in data:
                ctx.load_verify_locations(cafile=str(p))
            else:  # .cer в двоичном формате (DER)
                ctx.load_verify_locations(cadata=data)
            return ctx
        if os.environ.get("GIGACHAT_VERIFY_SSL", "1").strip().lower() in ("0", "false", "no"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return ssl.create_default_context()

    def _get_token(self):
        if self._token and time.time() < self._exp - 60:
            return self._token
        if not self.auth_key:
            raise LLMError("Не задан GIGACHAT_AUTH_KEY в файле .env")
        key = self.auth_key
        if ":" in key:  # дали client_id:client_secret вместо готового ключа
            key = base64.b64encode(key.encode()).decode()
        headers = {"Authorization": f"Basic {key}", "RqUID": str(uuid.uuid4()),
                   "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        req = urllib.request.Request(self.OAUTH_URL, data=f"scope={self.scope}".encode(), headers=headers,
                                     method="POST")
        try:
            try:
                with urllib.request.urlopen(req, timeout=60, context=self._ctx) as r:
                    data = json.loads(r.read().decode("utf-8"))
            except urllib.error.URLError as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e) or self.CERT_FILE.exists():
                    raise
                print("  GigaChat использует сертификат Минцифры, которого нет в системе. Скачиваю его...")
                if not self._auto_cert():
                    raise
                os.environ.setdefault("GIGACHAT_CA_BUNDLE", str(self.CERT_FILE))
                self._ctx = self._ssl()
                with urllib.request.urlopen(req, timeout=60, context=self._ctx) as r:
                    data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"Не удалось получить токен GigaChat: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}")
        except urllib.error.URLError as e:
            raise LLMError(f"Не удалось подключиться к GigaChat: {e}. Если ошибка про сертификат (CERTIFICATE_VERIFY_FAILED), "
                           f"см. README, раздел «Сертификат Минцифры».")
        self._token = data["access_token"]
        self._exp = data.get("expires_at", 0) / 1000 if data.get("expires_at") else time.time() + 1500
        return self._token

    def _headers(self):
        return {"Authorization": f"Bearer {self._get_token()}", "Content-Type": "application/json",
                "Accept": "application/json"}

    def _call(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens}, ensure_ascii=False).encode("utf-8")
        data = self._post(self.API_URL, self._headers(), body, ctx=self._ctx)
        u = data.get("usage", {})
        return strip_reasoning(data["choices"][0]["message"]["content"]), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


class OpenAICompatible(BaseLLM):
    provider = "openai"

    def __init__(self, model=None, **kw):
        super().__init__(model or os.environ.get("OPENAI_MODEL", ""), **kw)
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.scheme = os.environ.get("OPENAI_AUTH_SCHEME", "Bearer")
        extra = os.environ.get("OPENAI_EXTRA_BODY", "").strip()
        self.extra = json.loads(extra) if extra else {}

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"{self.scheme} {self.key}"
        return h

    def _call(self, messages):
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, **self.extra}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        data = self._post(f"{self.base}/chat/completions", self._headers(), body)
        u = data.get("usage", {}) or {}
        msg = data["choices"][0]["message"]
        return strip_reasoning(msg.get("content") or ""), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def make_llm(provider=None, model=None, **kw) -> BaseLLM:
    load_env()
    provider = (provider or os.environ.get("LLM_PROVIDER", "gigachat")).lower()
    if "temperature" not in kw and os.environ.get("LLM_TEMPERATURE"):
        kw["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    if "max_tokens" not in kw and os.environ.get("LLM_MAX_TOKENS"):
        kw["max_tokens"] = int(os.environ["LLM_MAX_TOKENS"])
    if provider == "gigachat":
        return GigaChat(model=model, **kw)
    if provider in ("openai", "yandex", "ollama"):
        return OpenAICompatible(model=model, **kw)
    raise LLMError(f"Неизвестный провайдер {provider}")
