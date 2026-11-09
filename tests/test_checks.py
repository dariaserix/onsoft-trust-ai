"""Автотесты проверок на настоящих документах (без обращения к модели).

Запуск: python -m unittest discover tests
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trustai import checks as C  # noqa: E402
from trustai.docs import load_package, split_sentences  # noqa: E402
from trustai.llm import BaseLLM  # noqa: E402
from trustai.pipeline import Solver  # noqa: E402

PK = ROOT / "data" / "train" / "packages"


class ScriptedLLM(BaseLLM):
    """Фальшивая модель для тестов: возвращает заранее заданные ответы по очереди."""
    provider = "scripted"

    def __init__(self, replies):
        super().__init__("test", use_cache=False, journal=ROOT / "results" / "test_journal.jsonl")
        self.replies = list(replies)

    def _call(self, messages, think=True):
        r = self.replies.pop(0) if self.replies else '{"status": "не знаю"}'
        return (r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)), 0, 0


def solve(pkg, q, replies, **cfg):
    return Solver(ScriptedLLM(replies), dict({"discrepancy_llm": False}, **cfg)).solve("T", q, load_package(PK / pkg))


class TestChecks(unittest.TestCase):
    def test_fake_quote_rejected(self):
        r = solve("P01", "Какой размер обеспечения заявки, в рублях?",
                  [{"status": "ответ", "answer": "25000", "mentions": [{"quote": "обеспечение заявки 25 000 рублей", "value": "25000"}]}],
                  max_retries=0, absent_check=False)
        self.assertEqual(r.status, "не знаю")

    def test_retry_fixes(self):
        r = solve("P01", "Какой размер обеспечения заявки, в рублях?",
                  [{"status": "ответ", "answer": "25000", "mentions": [{"quote": "обеспечение заявки 25 000 рублей", "value": "25000"}]},
                   {"status": "ответ", "answer": "18000", "mentions": [{"quote": "что составляет 18 000,00 рублей", "value": "18000"}]}],
                  max_retries=2)
        self.assertEqual((r.status, r.answer), ("ответ", "18000"))

    def test_number_not_in_quote(self):
        r = solve("P01", "Какой размер обеспечения заявки, в рублях?",
                  [{"status": "ответ", "answer": "19000", "mentions": [{"quote": "что составляет 18 000,00 рублей", "value": "19000"}]}],
                  max_retries=0)
        self.assertEqual(r.status, "не знаю")

    def test_amendment_old_value(self):
        pkg = load_package(PK / "P01")
        old = next(s for s in pkg.sentences if "опыт исполнения не менее 2" in s.text and s.file.startswith("01"))
        r = solve("P01", "Сколько исполненных контрактов нужно для подтверждения опыта и какая минимальная цена каждого, в рублях?",
                  [{"status": "ответ", "answer": "2", "mentions": [{"sid": old.sid, "quote": old.text, "value": "2"}]}],
                  max_retries=0)
        self.assertEqual(r.status, "не знаю")

    def test_amendment_both_quotes_new_answer_ok(self):
        """Модель привела и старую, и новую редакцию, но ответила по новой — это не обман."""
        pkg = load_package(PK / "P01")
        old = next(s for s in pkg.sentences if "24.11.2026 10:00" in s.text and s.file.startswith("01"))
        new = next(s for s in pkg.sentences if "27.11.2026 10:00" in s.text and pkg.docs[s.file].is_amendment)
        q = "До какой даты и времени принимаются заявки?"
        mentions = [{"sid": old.sid, "quote": old.text, "value": "24.11.2026 10:00"},
                    {"sid": new.sid, "quote": "Дата и время окончания срока подачи заявок: 27.11.2026 10:00", "value": "27.11.2026 10:00"}]
        r = solve("P01", q, [{"status": "ответ", "answer": "27.11.2026 10:00", "mentions": mentions}], max_retries=0)
        self.assertEqual((r.status, r.answer), ("ответ", "27.11.2026 10:00"))
        self.assertIn("27.11.2026", r.quote)
        # тот же набор цитат, но ответ по старой редакции — ловим
        r = solve("P01", q, [{"status": "ответ", "answer": "24.11.2026 10:00", "mentions": mentions}], max_retries=0)
        self.assertEqual(r.status, "не знаю")

    def test_content_answer_not_replaced_by_negation(self):
        """Содержательный ответ со словами «не установлено» не превращается в «нет, не установлено»."""
        a = "Лицензия на техническую защиту конфиденциальной информации; лицензия ФСБ России не требуется"
        self.assertEqual(C.canonical_yesno(a, "Какие лицензии должны быть у участника?"), a)
        self.assertEqual(C.canonical_yesno("Не установлено", "Какой размер обеспечения гарантии?"), "нет, не установлено")

    def test_stubborn_absent_not_accepted(self):
        """Перепроверка нашла ответ, а модель упорно повторяет «нет в документах» — это не принимается."""
        q = "Когда истекает срок подачи заявок (дата и время)?"
        absent = {"status": "нет в документах", "answer": "", "mentions": []}
        found = {"found": True, "file": "05_izmeneniya_izveshcheniya.txt",
                 "quote": "Дата и время окончания срока подачи заявок: 27.11.2026 10:00", "value": "27.11.2026 10:00"}
        r = solve("P01", q, [absent, found, absent, absent], max_retries=2)
        self.assertEqual((r.status, r.answer), ("ответ", "27.11.2026 10:00"))

    def test_negation_about_other_parameter(self):
        """«Не требуется» про обеспечение гарантийных обязательств — не ответ на вопрос о гарантийном сроке."""
        pkg = load_package(PK / "P02")
        s = next(x for x in pkg.sentences if "гарантийных обязательств не требуется" in x.text)
        ans = {"status": "ответ", "answer": "нет, не требуется", "mentions": [{"sid": s.sid, "quote": s.text, "value": "не требуется"}]}
        other = {"same_parameter": False, "reason": "цитата об обеспечении гарантийных обязательств"}
        r = solve("P02", "Какой гарантийный срок на результаты работ, в месяцах?", [ans, other], max_retries=0)
        self.assertEqual(r.status, "не знаю")
        r = solve("P02", "Какой гарантийный срок на результаты работ, в месяцах?", [ans, {"same_parameter": True}],
                  max_retries=0)
        self.assertEqual((r.status, r.answer), ("ответ", "нет, не установлено"))

    def test_number_written_in_words(self):
        """«в течение пяти рабочих дней» подтверждает ответ «5»; «тридцать шесть» — это 36, а не 306."""
        from trustai.textutil import words_to_digits
        self.assertEqual(words_to_digits("в течение пяти рабочих дней"), "в течение 5 рабочих дней")
        self.assertEqual(words_to_digits("36 (тридцать шесть) месяцев"), "36 (36) месяцев")
        self.assertEqual(words_to_digits("семья и тридцатилетие"), "семья и тридцатилетие")
        m = C.Mention(sid="", file="f", quote="", value="", evidence="в течение пяти рабочих дней с даты запроса")
        self.assertEqual(C.check_numbers("5", [m], question="За сколько рабочих дней заказчик предоставляет информацию?"), [])
        self.assertTrue(C.check_numbers("6", [m]))

    def test_table_and_header_retrieved(self):
        """Строка таблицы тянет за собой всю таблицу; шапка документа (заказчик, объект) всегда видна модели."""
        from trustai.retrieval import Retriever
        R = Retriever(load_package(PK / "P01"))
        sel = " ".join(s.text for s in R.select("Сколько этапов работ предусмотрено описанием объекта закупки?"))
        for row in ("1 | Обследование", "2 | Выполнение работ", "3 | Приёмка"):
            self.assertIn(row, sel)
        self.assertIn("Заказчик:", " ".join(s.text for s in R.select("Кто является заказчиком?")))
        # число в начале строки таблицы — значение ячейки, а не номер пункта
        m = C.Mention(sid="", file="f", quote="", value="", evidence="3 | Приёмка | Документ о приёмке")
        self.assertEqual(C.check_numbers("3", [m]), [])

    def test_excluded_is_negation_but_exclusively_is_not(self):
        """«Пункт … исключён» — требования нет; «исключительно СМП» и «за исключением случаев» — не отрицание."""
        self.assertTrue(C.has_negation("Пункт 3.4 (дополнительные требования к опыту участника) исключён"))
        self.assertTrue(C.has_negation("требование исключено"))
        self.assertFalse(C.has_negation("участие принимают исключительно субъекты малого предпринимательства"))
        self.assertFalse(C.has_negation("не допускается, за исключением случаев, предусмотренных законом"))

    def test_discrepancy_detected(self):
        r = solve("P01", "Какой срок выполнения работ, в календарных днях?",
                  [{"status": "ответ", "answer": "60", "mentions": [
                      {"quote": "выполняет работы в течение 60 календарных дней", "value": "60"},
                      {"quote": "семьдесят пять (75) календарных дней", "value": "75"}]}], max_retries=0)
        self.assertEqual(r.status, "расхождение")
        self.assertIn("60", r.answer)
        self.assertIn("75", r.answer)

    def test_calc_in_code(self):
        r = solve("P01", "Сколько рублей составит обеспечение исполнения контракта, если контракт заключат по начальной (максимальной) цене?",
                  [{"status": "ответ", "answer": "900000", "calc": "1800000*5/100", "mentions": [
                      {"quote": "Начальная (максимальная) цена контракта: 1 800 000,00 руб", "value": "1800000"},
                      {"quote": "5 % от цены контракта", "value": "5 %"}]}], max_retries=0)
        self.assertEqual(r.answer.split()[0], "90000")

    def test_negation_yes(self):
        r = solve("P04", "Нужна ли участнику лицензия ФСБ России? Ответьте да или нет.",
                  [{"status": "ответ", "answer": "да", "mentions": [
                      {"quote": "Требование о наличии лицензии ФСБ России к участникам настоящей закупки не установлено", "value": "да"}]}],
                  max_retries=0)
        self.assertEqual(r.status, "не знаю")

    def test_negation_no_ok(self):
        r = solve("P04", "Нужна ли участнику лицензия ФСБ России? Ответьте да или нет.",
                  [{"status": "ответ", "answer": "нет", "mentions": [
                      {"quote": "Требование о наличии лицензии ФСБ России к участникам настоящей закупки не установлено", "value": "нет"}]}],
                  max_retries=0)
        self.assertEqual(r.status, "ответ")
        self.assertTrue(r.answer.startswith("нет"))

    def test_safe_eval(self):
        self.assertAlmostEqual(C.safe_eval("1 800 000,00 * 5%"), 90000)
        self.assertIsNone(C.safe_eval("__import__('os')"))

    def test_split_abbrev(self):
        s = split_sentences("Работы выполняются по адресу: г. Образцово, ул. Больничная, д. 2. Далее текст.")
        self.assertEqual(len(s), 2)


if __name__ == "__main__":
    unittest.main()


class TestUnits(unittest.TestCase):
    def test_percent_instead_of_rubles(self):
        r = solve("P01", "Какой размер обеспечения заявки, в рублях?",
                  [{"status": "ответ", "answer": "1", "mentions": [{"quote": "обеспечение заявки в размере 1 % от НМЦК", "value": "1"}]}],
                  max_retries=0)
        self.assertEqual(r.status, "не знаю")

    def test_clause_number_not_a_value(self):
        r = solve("P01", "Какой размер обеспечения заявки, в рублях?",
                  [{"status": "ответ", "answer": "2.1", "mentions": [{"quote": "2.1. Участник предоставляет обеспечение заявки", "value": "2.1"}]}],
                  max_retries=0)
        self.assertEqual(r.status, "не знаю")
