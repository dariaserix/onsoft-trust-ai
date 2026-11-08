"""Агентный цикл: ответ модели → проверки кодом → (ИИ-критик) → исправление → снова проверки.

Если проверки так и не пройдены за отведённое число попыток, система честно отвечает «не знаю».
Какие проверки включены, задаёт конфиг (configs/*.json): так мы меряем вклад каждой.
"""
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from . import checks as C
from .docs import Package
from .llm import BaseLLM, ROOT
from .retrieval import Retriever, render_fragments
from .textutil import fmt_number, norm, normalize_answer_numbers, numbers, stems

PROMPTS = ROOT / "prompts"

DEFAULTS = {
    "mode": "pipeline",          # baseline | pipeline
    "retrieval": True,           # давать модели только нужные куски документа
    "top_k": 12, "per_file": 3, "max_fragment_chars": 9000,
    "quote_check": True,         # цитата дословно есть в документе
    "quote_repair": True,        # если модель чуть исказила цитату, взять точный текст фрагмента
    "number_check": True,        # числа из ответа есть в цитатах
    "calc_in_code": True,        # расчёты делает код, а не модель
    "amendment_check": True,     # значение не из старой редакции (изменения извещения)
    "discrepancy_check": True,   # разные значения в разных документах → «расхождение»
    "discrepancy_llm": True,     # второй запрос подтверждает, что это один и тот же параметр
    "yesno_check": True,         # «да/нет» и отрицания («не установлено»)
    "absent_check": True,        # «нет в документах» перепроверяется по полному тексту
    "absent_window_chars": 40000,
    "llm_critic": False,         # ИИ-критик проверяет итоговый ответ
    "max_retries": 2,            # попыток исправления после замечаний (0 = сразу «не знаю»)
    "no_think_fallback": True,   # пустой ответ (рассуждения упёрлись в лимит) → повтор без рассуждений
}


def load_prompt(name):
    return (PROMPTS / name).read_text(encoding="utf-8")


def fill(tpl: str, **kw):
    for k, v in kw.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    return tpl


@dataclass
class Result:
    id: str
    status: str
    answer: str = ""
    quote: str = ""
    file: str = ""
    trace: List[dict] = field(default_factory=list)

    def row(self):
        return {"id": self.id, "status": self.status, "answer": self.answer, "quote": self.quote, "file": self.file}


class Solver:
    def __init__(self, llm: BaseLLM, cfg: Optional[Dict] = None):
        self.llm = llm
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self._retrievers = {}

    def retriever(self, pkg: Package) -> Retriever:
        if pkg.name not in self._retrievers:
            self._retrievers[pkg.name] = Retriever(pkg)
        return self._retrievers[pkg.name]

    # ---------------------------------------------------------------- baseline
    def solve_baseline(self, qid, question, pkg: Package) -> Result:
        text = "\n\n".join(f"=== {n} ===\n{d.text}" for n, d in pkg.docs.items())
        msgs = [{"role": "system", "content": load_prompt("baseline_system.txt")},
                {"role": "user", "content": fill(load_prompt("baseline_user.txt"), text=text, question=question)}]
        r = self.llm.chat(msgs, purpose="baseline", qid=qid)
        data = C.parse_json(r.text) or {}
        st = C.norm_status(str(data.get("status", ""))) or ("ответ" if data.get("answer") else "")
        res = Result(qid, st or "", str(data.get("answer", "")), str(data.get("quote", "")), str(data.get("file", "")))
        res.trace.append({"step": "baseline", "raw": r.text[:2000]})
        return res

    # ---------------------------------------------------------------- pipeline
    def solve(self, qid, question, pkg: Package) -> Result:
        if self.cfg["mode"] == "baseline":
            return self.solve_baseline(qid, question, pkg)
        cfg = self.cfg
        trace = []
        if cfg["retrieval"]:
            frags = self.retriever(pkg).select(question, top_k=cfg["top_k"], per_file=cfg["per_file"],
                                               max_chars=cfg["max_fragment_chars"])
        else:
            frags = [s for s in pkg.sentences if not s.injection]
        trace.append({"step": "retrieval", "sids": [s.sid for s in frags], "chars": sum(len(s.text) for s in frags)})
        if pkg.injections:
            trace.append({"step": "injection", "caught": True, "critic": "injection",
                          "message": f"В документах найдено {len(pkg.injections)} посторонних инструкций, они вырезаны: "
                                     + " | ".join(s.text[:80] for s in pkg.injections[:3])})
        if not frags:
            return Result(qid, "не знаю", trace=trace + [{"step": "final", "message": "документы пустые или не читаются"}])

        msgs = [{"role": "system", "content": load_prompt("answer_system.txt")},
                {"role": "user", "content": fill(load_prompt("answer_user.txt"), package=pkg.name,
                                                 fragments=render_fragments(frags), question=question)}]
        absent_checked = False
        first_answer = None
        for attempt in range(cfg["max_retries"] + 1):
            r = self.llm.chat(msgs, purpose=f"answer#{attempt}", qid=qid)
            step = {"step": f"answer#{attempt}", "raw": r.text[:1500]}
            if not r.text.strip() and cfg.get("no_think_fallback", True):
                # модель так долго рассуждала, что не успела ответить: просим ответить без рассуждений
                r = self.llm.chat(msgs, purpose=f"answer#{attempt}-nothink", qid=qid, think=False)
                step.update(raw=r.text[:1500], no_think=True)
            data = C.parse_json(r.text)
            trace.append(step)
            if data is None:
                issues, cand, events = [C.Issue("format", "Ответ не в формате JSON. Верни только JSON.")], None, []
            else:
                cand, issues, events = self.evaluate(qid, question, pkg, data)
                if first_answer is None:
                    first_answer = {"status": data.get("status"), "answer": data.get("answer")}
                # «нет в документах» перепроверяем по полному тексту
                if cand is not None and cand.status == "нет в документах" and cfg["absent_check"] and not absent_checked:
                    absent_checked = True
                    found = self.absent_check(qid, question, pkg)
                    if found:
                        issues.append(C.Issue("absent", f"Ты ответил «нет в документах», но в документе {found['file']} "
                                                        f"есть: «{found['quote']}». Перепроверь и процитируй."))
                        events.append({"critic": "absent", "caught": True,
                                       "message": f"Модель сказала «нет в документах», а проверка нашла: «{found['quote'][:80]}»."})
                        cand = None
                # ИИ-критик
                if cand is not None and not issues and cfg["llm_critic"] and cand.status in ("ответ", "расхождение"):
                    prob = self.llm_critic(qid, question, cand)
                    if prob:
                        issues.append(C.Issue("llm_critic", f"Проверяющий нашёл проблему: {prob}"))
                        events.append({"critic": "llm_critic", "caught": True, "message": prob})
            step["issues"] = [{"critic": i.critic, "message": i.message} for i in issues]
            step["events"] = events
            if cand is not None and not issues:
                cand.trace = trace + [{"step": "final", "status": cand.status, "answer": cand.answer,
                                       "first_answer": first_answer}]
                return cand
            if attempt < cfg["max_retries"]:
                msgs = msgs + [{"role": "assistant", "content": r.text},
                               {"role": "user", "content": fill(load_prompt("retry_user.txt"),
                                                                issues="\n".join(f"- {i.message}" for i in issues))}]
        trace.append({"step": "final", "status": "не знаю", "first_answer": first_answer,
                      "message": "проверки не пройдены, система честно отвечает «не знаю»"})
        return Result(qid, "не знаю", trace=trace)

    # ------------------------------------------------------------- проверки
    def evaluate(self, qid, question, pkg, data):
        """Возвращает (кандидат-ответ или None, замечания, события)."""
        cfg = self.cfg
        issues, events = [], []
        status = C.norm_status(str(data.get("status", "")))
        answer = str(data.get("answer", "") or "").strip()
        raw_mentions = data.get("mentions") or []
        if status is None:
            return None, [C.Issue("format", "status должен быть одним из: ответ, расхождение, нет в документах, не знаю.")], events
        if status == "не знаю":
            return Result(qid, "не знаю"), [], events
        if status == "нет в документах":
            return Result(qid, "нет в документах"), [], events

        # 1. цитаты
        if cfg["quote_check"]:
            mentions, iss, ev = C.verify_mentions(pkg, raw_mentions, repair=cfg["quote_repair"])
            issues += iss
            events += ev
        else:
            mentions = []
            for rm in raw_mentions:
                if isinstance(rm, dict):
                    m = C.Mention(sid=str(rm.get("sid", "")), file=str(rm.get("file", "")), quote=str(rm.get("quote", "")),
                                  value=str(rm.get("value", "")))
                    m.evidence = m.out_quote = m.quote
                    mentions.append(m)
        if not mentions:
            issues.append(C.Issue("quote", "Нет ни одной подтверждённой цитаты. Ответ без цитаты не принимается."))
            return None, issues, events

        # 2. изменения извещения
        if cfg["amendment_check"]:
            iss, ev = C.check_amendments(pkg, mentions, answer + " " + str(data.get("calc", "") or ""))
            issues += iss
            events += ev
            active = [m for m in mentions if not m.superseded_by]
        else:
            active = mentions

        # 3. расчёт кодом
        calc_value = None
        if cfg["calc_in_code"]:
            expr = str(data.get("calc", "") or "").strip()
            if expr:
                calc_value = C.safe_eval(expr)
                if calc_value is None:
                    issues.append(C.Issue("calc", f"Не удалось вычислить выражение «{expr}». Запиши его как 1800000*5/100."))
                else:
                    ev_nums = []
                    for m in active:
                        ev_nums += numbers(C.strip_clause(m.evidence))
                    bad = [x for x in C.calc_operands(expr) if x not in (100.0,) and not any(abs(x - y) < 0.01 for y in ev_nums)]
                    if bad and cfg["number_check"]:
                        issues.append(C.Issue("calc", "В расчёте есть числа, которых нет в цитатах: "
                                                      + ", ".join("%g" % b for b in bad) + ". Приведи цитаты с ними."))
                    unit = " руб." if "руб" in norm(question) else ""
                    old = answer
                    answer = fmt_number(round(calc_value, 2)) + unit
                    # «поймано» — только если модель сама назвала другой результат (а не просто записала формулу)
                    own = [x for x in numbers(old) if not any(abs(x - y) < 0.01 for y in C.calc_operands(expr))]
                    if own and not any(abs(x - calc_value) < 0.01 for x in own):
                        events.append({"critic": "calc", "caught": True,
                                       "message": f"Модель посчитала «{old}», код по её же выражению {expr} получил {answer}."})
            elif C.is_calc(question) and C.check_numbers(answer, active):
                issues.append(C.Issue("calc", "Это вопрос на расчёт. Не считай сам: запиши выражение в поле calc "
                                              "и дай цитаты со всеми числами."))

        # 4. расхождение между документами (для расчётов не применяем: там числа из разных пунктов)
        if cfg["discrepancy_check"] and calc_value is None:
            status, answer, active, iss, ev = self.resolve_discrepancy(qid, question, status, answer, active)
            issues += iss
            events += ev

        # 5. числа из ответа есть в цитатах
        if cfg["number_check"] and status == "ответ":
            issues += C.check_numbers(answer, active, calc_value, question)

        # 6. да/нет и отрицания
        if cfg["yesno_check"] and status == "ответ":
            issues += C.check_yesno(question, answer, active)
            answer = C.canonical_yesno(answer, question)

        if not answer:
            issues.append(C.Issue("format", "Поле answer пустое."))
        # итоговая цитата: предпочитаем фрагмент из изменений и тот, где есть число ответа
        main = self.pick_main(active or mentions, answer)
        quote = main.out_quote or main.quote
        if cfg["quote_check"] and not pkg.quote_in_docs(quote):
            issues.append(C.Issue("quote", "Итоговая цитата не найдена в документах."))
        res = Result(qid, status, normalize_answer_numbers(answer), quote, main.file)
        # все проверенные цитаты — для ИИ-критика (одной итоговой цитаты мало: в расчёте их две)
        res.evidence = [(m.file, m.evidence or m.out_quote) for m in (active or mentions)]
        if calc_value is not None:
            res.evidence.append(("расчёт кодом", f"{data.get('calc')} = {fmt_number(round(calc_value, 2))}"))
        return res, issues, events

    def pick_main(self, mentions, answer):
        ans = numbers(answer)
        ans_words = set(stems(answer))

        def score(m):
            sc = 0
            if m.amendment:
                sc += 2
            if ans and any(any(abs(a - x) < 0.01 for x in numbers(m.evidence)) for a in ans):
                sc += 3
            # при равенстве — цитата, которая по словам ближе всего к ответу
            if ans_words:
                sc += len(ans_words & set(stems(m.evidence or m.quote))) / len(ans_words)
            return sc
        return sorted(mentions, key=score, reverse=True)[0]

    def resolve_discrepancy(self, qid, question, status, answer, mentions):
        issues, events = [], []
        amend = [m for m in mentions if m.amendment]
        if amend:
            if status == "расхождение":
                events.append({"critic": "discrepancy", "caught": True,
                               "message": "Модель увидела расхождение, но это изменение извещения: верна новая редакция."})
                status = "ответ"
                answer = amend[0].value or answer
            return status, answer, mentions, issues, events
        by_file = {}
        for m in mentions:
            vals = tuple(sorted(set(round(x, 2) for x in numbers(m.value))))
            if vals:
                by_file.setdefault(m.file, set()).add(vals)
        distinct = set()
        for vs in by_file.values():
            distinct |= vs
        if len(by_file) >= 2 and len(distinct) >= 2:
            confirmed, values, keep = True, None, None
            if self.cfg["discrepancy_llm"]:
                confirmed, values, keep = self.discrepancy_llm(qid, question, mentions)
            if confirmed:
                vals = values or [m.value for m in mentions]
                uniq = []
                for v in vals:
                    if v and v not in uniq:
                        uniq.append(v)
                if status != "расхождение":
                    events.append({"critic": "discrepancy", "caught": True,
                                   "message": "Модель дала одно значение, но документы противоречат друг другу: "
                                              + " / ".join(f"{m.file}: {m.value}" for m in mentions)})
                return "расхождение", " и ".join(uniq[:3]), mentions, issues, events
            if keep:
                mentions = [m for m in mentions if m.sid in keep] or mentions
            if status == "расхождение":
                events.append({"critic": "discrepancy", "caught": True,
                               "message": "Модель нашла «расхождение», но цитаты о разных параметрах."})
                issues.append(C.Issue("discrepancy", "Цитаты относятся к разным параметрам, противоречия нет. "
                                                     "Дай одно значение для параметра из вопроса."))
            return status, answer, mentions, issues, events
        if status == "расхождение":
            events.append({"critic": "discrepancy", "caught": True,
                           "message": "Модель заявила расхождение, но разных значений в цитатах нет."})
            if len(distinct) == 1 and len(by_file) >= 1:
                status = "ответ"
                answer = mentions[0].value or answer
            else:
                issues.append(C.Issue("discrepancy", "Для «расхождения» нужны цитаты из двух документов с разными значениями."))
        return status, answer, mentions, issues, events

    # ------------------------------------------------------------- доп. запросы к модели
    def discrepancy_llm(self, qid, question, mentions):
        quotes = "\n".join(f"[{m.sid or '-'}] ({m.file}) {m.evidence or m.quote}" for m in mentions)
        msgs = [{"role": "system", "content": load_prompt("discrepancy_system.txt")},
                {"role": "user", "content": f"Вопрос: {question}\nЦитаты:\n{quotes}"}]
        r = self.llm.chat(msgs, purpose="discrepancy", qid=qid)
        d = C.parse_json(r.text) or {}
        if d.get("contradiction") is True:
            vals = [str(v) for v in (d.get("values") or []) if str(v).strip()]
            return True, vals or None, None
        keep = [str(x).upper() for x in (d.get("relevant_sids") or [])]
        return False, None, keep or None

    def absent_check(self, qid, question, pkg: Package):
        text = pkg.full_text()
        w = self.cfg["absent_window_chars"]
        parts = [text[i:i + w] for i in range(0, len(text), w)] or [""]
        for k, part in enumerate(parts, 1):
            msgs = [{"role": "system", "content": load_prompt("absent_system.txt")},
                    {"role": "user", "content": fill(load_prompt("absent_user.txt"), package=pkg.name,
                                                     part=f"{k} из {len(parts)}", text=part, question=question)}]
            r = self.llm.chat(msgs, purpose="absent_check", qid=qid)
            d = C.parse_json(r.text) or {}
            if d.get("found") is True:
                q = str(d.get("quote", ""))
                f = pkg.quote_in_docs(q)
                if f:  # засчитываем только проверенную цитату
                    return {"file": f, "quote": q, "value": d.get("value", "")}
        return None

    def llm_critic(self, qid, question, cand: Result):
        msgs = [{"role": "system", "content": load_prompt("critic_system.txt")},
                {"role": "user", "content": fill(load_prompt("critic_user.txt"), question=question, answer=cand.answer,
                                                 status=cand.status,
                                                 quotes="\n".join(f"- ({f}) {q}" for f, q in
                                                                  (getattr(cand, "evidence", None) or [(cand.file, cand.quote)])))}]
        r = self.llm.chat(msgs, purpose="llm_critic", qid=qid)
        d = C.parse_json(r.text) or {}
        if d.get("ok") is False:
            return str(d.get("problem", "ответ не подтверждается цитатами"))[:300]
        return None
