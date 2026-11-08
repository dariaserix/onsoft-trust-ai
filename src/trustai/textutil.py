"""Нормализация текста и чисел.

Функции norm(), numbers() и decode_like_checker() повторяют логику check.py
один в один. Так мы проверяем цитаты и числа ровно так же, как их проверит жюри.
"""
import re

_SPACES = "  "


def norm(s):
    s = (s or "").lower().replace("ё", "е").replace(" ", " ").replace(" ", " ")
    s = re.sub("[«»„“”\"']", '"', s)
    s = re.sub("[–—−]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


_NUM_RE = re.compile(r"\d+(?:[   ]\d{3})*(?:[.,]\d+)?")


def numbers(s):
    out = []
    for m in _NUM_RE.findall(s or ""):
        try:
            out.append(float(re.sub(r"[   ]", "", m).replace(",", ".")))
        except ValueError:
            pass
    return out


def decode_like_checker(raw: bytes) -> str:
    """Так читает файлы check.py: сначала utf-8, потом cp1251."""
    for enc in ("utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def fmt_number(x: float) -> str:
    """18000.0 -> '18000', 2.5 -> '2.5'."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return ("%.2f" % x).rstrip("0").rstrip(".")


def numbers_in(haystack_numbers, needles, tol=0.01):
    """Все ли числа needles встречаются среди haystack_numbers."""
    return all(any(abs(h - n) < tol for h in haystack_numbers) for n in needles)


def normalize_answer_numbers(answer: str) -> str:
    """'1 800 000,00 руб.' -> '1800000 руб.' (числа цифрами, без пробелов)."""
    def repl(m):
        raw = m.group(0)
        try:
            v = float(re.sub(r"[   ]", "", raw).replace(",", "."))
        except ValueError:
            return raw
        # даты/время не трогаем: у них точки и двоеточия обрабатываются отдельно
        return fmt_number(v)

    # не трогаем даты вида 27.11.2026 и время 10:00
    parts = re.split(r"(\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}:\d{2})", answer or "")
    out = []
    for p in parts:
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}:\d{2}", p or ""):
            out.append(p)
        else:
            out.append(_NUM_RE.sub(repl, p))
    return "".join(out)


# --- простая морфология для поиска -------------------------------------------

STOP = set("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее если есть еще же за
здесь и из или им их к как какая какие каким каких какой какое ко когда кто ли либо мне может мы на над надо наш не него нее нет
ни них но ну о об однако он она они оно от очень по под при с со так также такой там те тем то того тоже той только том ты у уже
хотя чего чей чем что чтобы чье чья эта эти это я ответьте ответ сколько который которая которые должно должна должен должны
нужно нужна нужен случае рублях процентах днях месяцах
""".split())


def tokens(text):
    return re.findall(r"[a-zа-я0-9]+", norm(text))


_ENDINGS = sorted("ами ями ого его ому ему ыми ими ией иям иях ов ев ей ой ий ый ая яя ое ее ые ие ам ям ом ем ах ях "
                  "ую юю ок ек а я о е и ы у ю ь".split(), key=len, reverse=True)


def stem(w: str) -> str:
    """Грубая основа слова: отрезаем окончание и берём первые 4 буквы.
    Так «заявки/заявок», «время/времени» дают одну основу. Замер покрытия поиска (tests/retrieval_coverage.py):
    правильная цитата в выдаче — 61/61 на всех наборах (с основой из 5 букв было 58/61 на переформулировках)."""
    if w.isdigit():
        return w
    for e in _ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 4:
            w = w[:-len(e)]
            break
    return w[:4]


def stems(text, drop_stop=True):
    return [stem(t) for t in tokens(text) if not (drop_stop and t in STOP) and len(t) > 1]


# --- числа, написанные словами («в течение пяти рабочих дней», «тридцать шесть месяцев») ----------------

_UNITS = [("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14), ("пятнадцат", 15),
          ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18), ("девятнадцат", 19),
          ("один", 1), ("одн", 1), ("два", 2), ("две", 2), ("двух", 2), ("двум", 2), ("тр", 3), ("четыр", 4),
          ("пят", 5), ("шест", 6), ("сем", 7), ("восьм", 8), ("восем", 8), ("девят", 9), ("десят", 10)]
_TENS = [("двадцат", 20), ("тридцат", 30), ("сорок", 40), ("пятьдесят", 50), ("пятидесят", 50),
         ("шестьдесят", 60), ("шестидесят", 60), ("семьдесят", 70), ("семидесят", 70), ("восемьдесят", 80),
         ("восьмидесят", 80), ("девяност", 90)]
_HUNDREDS = [("двест", 200), ("двухсот", 200), ("трист", 300), ("трехсот", 300), ("четырест", 400),
             ("четырехсот", 400), ("пятьсот", 500), ("пятисот", 500), ("шестьсот", 600), ("семьсот", 700),
             ("восемьсот", 800), ("девятьсот", 900), ("сто", 100), ("ста", 100)]
# точные формы для коротких основ, чтобы «три» не путалось с «тридцать», «семь» — с «семья» и т.п.
_EXACT_UNITS = {
    "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1, "одному": 1, "одним": 1, "одну": 1, "одном": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "двумя": 2,
    "три": 3, "трех": 3, "трем": 3, "тремя": 3,
    "четыре": 4, "четырех": 4, "четырем": 4, "четырьмя": 4,
    "пять": 5, "пяти": 5, "пятью": 5, "шесть": 6, "шести": 6, "шестью": 6, "семь": 7, "семи": 7, "семью": 7,
    "восемь": 8, "восьми": 8, "восемью": 8, "девять": 9, "девяти": 9, "девятью": 9,
    "десять": 10, "десяти": 10, "десятью": 10,
}


def _word_value(w: str):
    """Значение одного числительного (или None) и разряд: h — сотни, t — десятки, u — единицы.
    Сравниваем с точными формами (основа + падежное окончание), чтобы «семья» или «тридцатилетие» не стали числом."""
    if w in _EXACT_UNITS:
        return _EXACT_UNITS[w], "u"
    if w in ("сто", "ста"):
        return 100, "h"
    for stem_, v in _HUNDREDS:
        if len(stem_) > 3 and re.fullmatch(stem_ + r"(и|а|ам|ами|ах)?", w):
            return v, "h"
    for stem_, v in _TENS:
        if re.fullmatch(stem_ + r"(ь|и|ью|а|о)?", w):
            return v, "t"
    for stem_, v in _UNITS[:9]:  # 11–19
        if re.fullmatch(stem_ + r"(ь|и|ью)", w):
            return v, "u"
    return None


def words_to_digits(text: str) -> str:
    """«в течение пяти рабочих дней» → «в течение 5 рабочих дней»; «тридцать шесть» → «36».
    Нужна проверкам: число из ответа модели ищется в цитате, а в документе оно может быть написано словами."""
    if not text:
        return text
    out, buf_val, buf_kinds = [], 0, ""
    parts = re.split(r"([а-яёА-ЯЁ]+)", text)

    def flush():
        nonlocal buf_val, buf_kinds
        if buf_kinds:
            out.append(str(buf_val))
        buf_val, buf_kinds = 0, ""

    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"[а-яёА-ЯЁ]+", p):
            r = _word_value(p.lower().replace("ё", "е"))
            # составное число: сотни → десятки → единицы, по убыванию разрядов
            if r and (not buf_kinds or "htu".index(r[1]) > "htu".index(buf_kinds[-1])):
                buf_val += r[0]
                buf_kinds += r[1]
                continue
            flush()
            if r:
                buf_val, buf_kinds = r[0], r[1]
                continue
            out.append(p)
        elif buf_kinds and p.strip() == "":
            continue  # пробел внутри составного числа
        else:
            flush()
            out.append(p)
    flush()
    res = "".join(out)
    # между числом и следующим словом пробел мог пропасть при склейке — вернём его
    return re.sub(r"(\d)([а-яёА-ЯЁ])", r"\1 \2", res)
