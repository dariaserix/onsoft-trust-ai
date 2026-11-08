"""Покрытие поиска: попала ли правильная цитата в фрагменты, которые видит модель. Без обращения к модели.

python tests/retrieval_coverage.py [data/train data/paraphrase ...]
Правильные цитаты берутся из answers_public.json — это инструмент разработчика (как check.py), решатель их не видит.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trustai.docs import load_package  # noqa: E402
from trustai.retrieval import Retriever  # noqa: E402
from trustai.runner import read_questions  # noqa: E402
from trustai.textutil import norm  # noqa: E402


def coverage(data_dir):
    data_dir = Path(data_dir)
    gold = {g["id"]: g for g in json.loads((data_dir / "answers_public.json").read_text(encoding="utf-8"))}
    pk, miss, total, chars = {}, [], 0, 0
    for q in read_questions(data_dir / "questions.csv"):
        g = gold.get(q["id"])
        if not g or not g.get("quote"):
            continue
        p = q["package"]
        if p not in pk:
            pk[p] = Retriever(load_package(data_dir / "packages" / p))
        sel = pk[p].select(q["question"])
        text = norm(" ".join(s.text for s in sel))
        total += 1
        chars += sum(len(s.text) for s in sel)
        if norm(g["quote"]) not in text:
            miss.append(q["id"])
    return total, miss, chars // max(total, 1)


if __name__ == "__main__":
    for d in sys.argv[1:] or ["data/train", "data/paraphrase"]:
        total, miss, avg = coverage(ROOT / d)
        print(f"{d}: цитата попала в выдачу {total - len(miss)}/{total}, в среднем {avg} символов; мимо: {' '.join(miss)}")
