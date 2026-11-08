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
