"""Поиск нужных кусков документа под вопрос (BM25 по основам слов + словарь синонимов закупок).

Зачем: модель видит 15-25 относящихся к делу предложений, а не 60 страниц.
Меньше шума, меньше выдумок, меньше токенов.
"""
import math
from collections import Counter
from typing import List

from .docs import Package, Sentence
from .textutil import stems

# Синонимы языка закупок: если в вопросе есть основа слева, добавляем слова справа.
SYNONYMS = {
    "начал": "нмцк начальная максимальная цена контракта",
    "максим": "нмцк начальная максимальная цена контракта",
    "нмцк": "начальная максимальная цена контракта",
    "обесп": "обеспечение размер",
    "гаран": "гарантия гарантийный гарантийные месяцев",
    "удале": "место выполнения адрес территория удаленно дистанционно присутствие",
    "удалё": "место выполнения адрес территория удаленно дистанционно присутствие",
    "выпол": "выполнения срок календарных",
    "где": "место адрес территория удаленно",
    "адрес": "место территория по адресу",
    "прини": "окончания срока подачи заявок принимаются",
    "заяв": "заявок заявки подачи",  # «заяв» ловит все формы: заявка, заявки, заявок (основа «заявк» не ловила «заявок»)
    "выпис": "выписка егрюл единого государственного реестра юридических лиц",
    "егрюл": "выписка единого государственного реестра юридических лиц",
    "прове": "приемка проверка рабочих дней документа о приемке",
    "приём": "приемка проверка рабочих дней документа о приемке",
    "приемк": "приемка проверка рабочих дней документа",
    "субпо": "субподрядчиков соисполнителей привлечь малого предпринимательства",
    "специ": "специалистов персонал сотрудников привлекает",
    "лицен": "лицензия лицензии фсб",
    "реест": "реестр российских программ российского программного обеспечения",
    "прогр": "программного обеспечения реестр российских программ",
    "опыт": "опыт исполненных договоров контрактов стоимостью",
    "испол": "исполненных договоров контрактов стоимостью",
    "малог": "малого предпринимательства смп сонко исключительно ограничение",
    "смп": "малого предпринимательства субъекты",
    "огран": "ограничение исключительно",
    "рабоч": "рабочих дней",
}

K1, B = 1.4, 0.6


class Retriever:
    def __init__(self, pkg: Package):
        self.pkg = pkg
        self.docs = [s for s in pkg.sentences if not s.injection]
        self.tf = [Counter(stems(s.text)) for s in self.docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs) or 1
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def expand(self, question: str) -> Counter:
        q = Counter(stems(question))
        extra = Counter()
        for t in list(q):
            for key, words in SYNONYMS.items():
                if t.startswith(key) or key.startswith(t) and len(t) >= 4:
                    for w in stems(words):
                        extra[w] += 0.5
        q.update(extra)
        return q

    def score(self, question: str) -> List[tuple]:
        q = self.expand(question)
        out = []
        for i, c in enumerate(self.tf):
            sc = 0.0
            for t, w in q.items():
                f = c.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                sc += w * idf * f * (K1 + 1) / (f + K1 * (1 - B + B * self.len[i] / self.avg))
            if sc > 0:
                out.append((sc, self.docs[i]))
        out.sort(key=lambda x: -x[0])
        return out

    def select(self, question: str, top_k=12, per_file=3, include_amendments=True, max_chars=9000) -> List[Sentence]:
        ranked = self.score(question)
        chosen, seen, seen_text = [], set(), set()

        def add(s):
            key = (s.file, s.text.strip().lower())
            if s.sid not in seen and key not in seen_text:  # дубли шаблонных фраз не показываем
                seen.add(s.sid)
                seen_text.add(key)
                chosen.append(s)

        for _, s in ranked[:top_k]:
            add(s)
        # по несколько лучших из каждого файла: чтобы заметить расхождение между документами
        per = Counter()
        for _, s in ranked:
            if per[s.file] < per_file:
                per[s.file] += 1
                add(s)
        # файл изменений извещения показываем целиком: он важнее старой редакции
        if include_amendments:
            for s in self.docs:
                if self.pkg.docs[s.file].is_amendment:
                    add(s)
        # укладываемся в бюджет символов, сохраняя порядок важности
        total, res = 0, []
        for s in chosen:
            if total + len(s.text) > max_chars and res:
                break
            total += len(s.text)
            res.append(s)
        # показываем в порядке документов: так модели проще читать
        order = {s.sid: i for i, s in enumerate(self.pkg.sentences)}
        res.sort(key=lambda s: order[s.sid])
        return res


def render_fragments(sentences: List[Sentence]) -> str:
    lines, cur = [], None
    for s in sentences:
        if s.file != cur:
            cur = s.file
            lines.append(f"\n### Файл {s.file}")
        lines.append(f"[{s.sid}] {s.text}")
    return "\n".join(lines).strip()
