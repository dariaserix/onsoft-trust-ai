"""Чтение документов закупки.

Устойчиво к трудным файлам:
- пустой файл: помечается, не ломает систему;
- другая кодировка (cp1251, utf-16, BOM): определяется автоматически;
- спрятанная инструкция для ИИ: такие предложения вырезаются из контекста и пишутся в журнал;
- огромный документ: модель видит только найденные куски (см. retrieval.py).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .textutil import decode_like_checker, norm


@dataclass
class Sentence:
    sid: str          # S1, S2, ... (уникален в пределах закупки)
    file: str
    text: str
    line: int
    injection: bool = False


@dataclass
class Doc:
    name: str
    text: str                 # как прочитали мы (правильная кодировка)
    checker_text: str         # как прочитает check.py (для проверки цитат)
    encoding: str
    empty: bool
    is_amendment: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class Amendment:
    old: str            # начало старой редакции пункта
    file: str           # в каком файле был пункт
    new: str            # новая редакция


@dataclass
class Package:
    name: str
    path: Path
    docs: Dict[str, Doc]
    sentences: List[Sentence]
    amendments: List[Amendment]
    injections: List[Sentence]
    warnings: List[str]

    def sentence(self, sid: str) -> Optional[Sentence]:
        return self._by_sid.get((sid or "").strip().upper())

    def __post_init__(self):
        self._by_sid = {s.sid: s for s in self.sentences}
        self._checker_norm = {n: norm(d.checker_text) for n, d in self.docs.items()}

    def quote_in_docs(self, quote: str) -> Optional[str]:
        """Проверка цитаты так же, как в check.py. Возвращает имя файла или None."""
        q = norm(quote)
        if len(q) < 8:
            return None
        for name, t in self._checker_norm.items():
            if q in t:
                return name
        return None

    def full_text(self, exclude_injections=True) -> str:
        parts = []
        for name, d in self.docs.items():
            if d.empty:
                parts.append(f"=== ФАЙЛ {name} (пустой) ===")
                continue
            body = d.text
            if exclude_injections:
                for s in self.injections:
                    if s.file == name:
                        body = body.replace(s.text, "[удалено: посторонняя инструкция]")
            parts.append(f"=== ФАЙЛ {name} ===\n{body}")
        return "\n\n".join(parts)


# --- кодировки ----------------------------------------------------------------

def smart_decode(raw: bytes):
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace"), "utf-8-sig"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    # cp1251 против koi8-r: выбираем ту, где больше частых русских букв
    best = None
    for enc in ("cp1251", "koi8-r", "cp866"):
        try:
            t = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        score = sum(t.count(c) for c in "оеаинтсрвлОЕАИНТСРВЛ")
        if best is None or score > best[2]:
            best = (t, enc, score)
    if best:
        return best[0], best[1]
    return raw.decode("utf-8", "replace"), "utf-8?"


# --- спрятанные инструкции ----------------------------------------------------

_INJECTION_RE = re.compile(
    r"(игнорир\w*|проигнорир\w*|забудь\w*|не обращай внимани\w*|ignore\s+(all|previous|the)|disregard|"
    r"system\s*prompt|системн\w+ промпт|"
    r"(для|к)\s+(ии|ai|нейросет\w*|ассистент\w*|языков\w+ модел\w*|chatgpt|gpt|гигачат\w*|gigachat)\b|"
    r"\b(ии|ai|нейросеть|ассистент|ии-помощник)\s*[,:!]|"
    r"(ответь|отвечай|напиши|укажи)\w*\s+(всегда|только|что)\b)",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


# --- разбиение на предложения -------------------------------------------------

_ABBR = {"г", "ул", "д", "руб", "коп", "т", "п", "пп", "им", "стр", "корп", "обл", "р", "тыс", "млн", "млрд",
         "ст", "ч", "рис", "см", "кв", "пом", "оф", "мкр", "пр", "пер", "наб", "ш", "е", "напр", "др", "т.е", "т.д",
         "т.п", "гг", "вкл", "мин", "сек", "max", "min", "no", "n", "№"}


def split_sentences(line: str) -> List[str]:
    out, start = [], 0
    for m in re.finditer(r"[.!?]\s+(?=[«\"(]?[А-ЯЁA-Z0-9])", line):
        before = line[start:m.start()]
        last = re.findall(r"([A-Za-zА-Яа-яЁё.]+)$", before)
        word = last[0].lower().rstrip(".") if last else ""
        # не режем после сокращений и после номеров пунктов вида "2.1"
        if word in _ABBR or re.fullmatch(r"\s*\d+(\.\d+)*", before):
            continue
        seg = line[start:m.start() + 1].strip()
        if seg:
            out.append(seg)
        start = m.end()
    tail = line[start:].strip()
    if tail:
        out.append(tail)
    return out


# --- изменения извещения ------------------------------------------------------

_AMEND_RE = re.compile(
    r"Пункт\s*«(?P<old>.+?)»\s*\((?P<file>[^)]+)\)\s*изложить в (?:следующей|новой) редакции:\s*«(?P<new>.+?)»\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_amendment_doc(name: str, text: str) -> bool:
    head = norm(text[:400])
    return "izmenen" in name.lower() or "изменения извещения" in head or "изменение извещения" in head


def parse_amendments(name: str, text: str) -> List[Amendment]:
    res = []
    for m in _AMEND_RE.finditer(text):
        old = m.group("old").rstrip("…").rstrip(".").strip()
        res.append(Amendment(old=old, file=m.group("file").strip(), new=m.group("new").strip()))
    return res


# --- загрузка -----------------------------------------------------------------

def load_package(path) -> Package:
    path = Path(path)
    docs, sentences, amendments, injections, warnings = {}, [], [], [], []
    files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md", "")])
    n = 0
    for f in files:
        try:
            raw = f.read_bytes()
        except OSError as e:
            warnings.append(f"{f.name}: не удалось прочитать ({e})")
            continue
        text, enc = smart_decode(raw)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        empty = not text.strip()
        d = Doc(name=f.name, text=text, checker_text=decode_like_checker(raw), encoding=enc, empty=empty,
                is_amendment=(not empty and is_amendment_doc(f.name, text)))
        if empty:
            d.warnings.append("пустой файл")
            warnings.append(f"{f.name}: пустой файл")
        if enc not in ("utf-8", "utf-8-sig"):
            d.warnings.append(f"кодировка {enc}")
            warnings.append(f"{f.name}: кодировка {enc}")
        docs[f.name] = d
        if empty:
            continue
        if d.is_amendment:
            amendments.extend(parse_amendments(f.name, text))
        for li, line in enumerate(text.split("\n"), 1):
            if not line.strip():
                continue
            for seg in split_sentences(line):
                n += 1
                s = Sentence(sid=f"S{n}", file=f.name, text=seg, line=li, injection=looks_like_injection(seg))
                if s.injection:
                    injections.append(s)
                sentences.append(s)
    if injections:
        warnings.append(f"найдено и вырезано посторонних инструкций: {len(injections)}")
    return Package(name=path.name, path=path, docs=docs, sentences=sentences, amendments=amendments,
                   injections=injections, warnings=warnings)
