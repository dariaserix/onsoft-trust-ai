"""Точка входа. Все команды запускаются из папки проекта.

  python run.py check                         проверить связь с моделью
  python run.py ask P01 "Какой срок выполнения работ?"   задать один вопрос и увидеть все проверки
  python run.py solve                         прогнать вопросы (по умолчанию data/train, конфиг configs/final.json)
  python run.py solve --data data/control     контрольный набор в день защиты → results/control/answers.csv
  python run.py ladder                        замер «было → стало» по всем шагам из configs/
  python run.py report                        собрать дашборд results/report.html
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trustai.docs import load_package  # noqa: E402
from trustai.llm import make_llm, LLMError  # noqa: E402
from trustai.pipeline import Solver  # noqa: E402
from trustai.runner import load_config, run_set, score  # noqa: E402

try:  # корректный вывод кириллицы в консоли Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def cmd_check(a):
    llm = make_llm(use_cache=False)
    print(f"Модель: {llm.name}")
    r = llm.chat([{"role": "user", "content": "Ответь одним словом: столица России?"}], purpose="check")
    print(f"Ответ: {r.text.strip()}  ({r.prompt_tokens}+{r.completion_tokens} токенов, {r.seconds:.1f} с)")
    print("Связь с моделью есть.")


def cmd_ask(a):
    llm = make_llm()
    cfg = load_config(a.config)
    data = Path(a.data)
    pkg = load_package(data / "packages" / a.package)
    res = Solver(llm, cfg).solve("ASK", a.question, pkg)
    print(f"\nВопрос: {a.question}")
    for st in res.trace:
        name = st.get("step")
        if name == "retrieval":
            print(f"  поиск: выбрано фрагментов {len(st['sids'])} ({st['chars']} символов)")
        elif name and name.startswith("answer"):
            print(f"  {name}: модель ответила")
            for ev in st.get("events", []):
                print(f"     ⚠ поймано [{ev['critic']}]: {ev['message']}")
            for i in st.get("issues", []):
                print(f"     ✗ проверка [{i['critic']}]: {i['message']}")
            if not st.get("issues"):
                print("     ✓ все проверки пройдены")
        elif name == "injection":
            print(f"  ⚠ {st['message']}")
    print(f"\nИТОГ: {res.status} | {res.answer}\nЦитата: {res.quote}\nФайл: {res.file}")


def cmd_solve(a):
    llm = make_llm()
    cfg = load_config(a.config)
    out = Path(a.out or (ROOT / "results" / ("control" if "control" in str(a.data) else "runs/" + cfg.get("name", "run"))))
    ans, st = run_set(llm, cfg, a.data, out, only=a.only, limit=a.limit)
    print(f"\nГотово: {ans}\nЗапросов к модели: {st['calls']} (из кэша {st['cached']}), токенов: "
          f"{st['prompt_tokens'] + st['completion_tokens']}, время: {st['wall_seconds']} с")
    sc = score(ans, a.data, out / "result.csv")
    if sc:
        print(sc["text"])
        (out / "score.json").write_text(json.dumps(sc["summary"], ensure_ascii=False), encoding="utf-8")


def cmd_ladder(a):
    llm = make_llm()
    configs = sorted((ROOT / "configs").glob("[0-9][0-9]_*.json"))
    if a.steps:
        configs = [c for c in configs if any(c.name.startswith(s) for s in a.steps)]
    rows = []
    for c in configs:
        cfg = load_config(c)
        print(f"\n===== {cfg['name']}: {cfg['title']} =====")
        out = ROOT / "results" / "runs" / cfg["name"]
        ans, st = run_set(llm, cfg, a.data, out, only=a.only, limit=a.limit, progress=False)
        sc = score(ans, a.data, out / "result.csv")
        s = sc["summary"] if sc else {}
        (out / "score.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        print(f"  индекс доверия {s.get('index')} | верно {s.get('correct')} | не знаю {s.get('careful')} | "
              f"обман {s.get('lies')} | запросов {st['calls']} | токенов {st['prompt_tokens'] + st['completion_tokens']}")
        rows.append({"step": cfg["name"], "title": cfg["title"], "description": cfg.get("description", ""), **s,
                     "calls": st["calls"], "tokens": st["prompt_tokens"] + st["completion_tokens"],
                     "seconds": st["wall_seconds"], "model": st["model"]})
    (ROOT / "results" / "ladder.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nСводка сохранена в results/ladder.json. Дашборд: python run.py report")


def cmd_report(a):
    from trustai.report import build_report
    p = build_report(ROOT / "results", Path(a.data))
    print(f"Дашборд: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("ask")
    p.add_argument("package")
    p.add_argument("question")
    for p in (p, sub.add_parser("solve"), sub.add_parser("ladder"), sub.add_parser("report")):
        p.add_argument("--data", default=str(ROOT / "data" / "train"))
        p.add_argument("--config", default=str(ROOT / "configs" / "final.json"))
        p.add_argument("--out")
        p.add_argument("--only", nargs="*")
        p.add_argument("--limit", type=int)
        p.add_argument("--steps", nargs="*")
    a = ap.parse_args()
    try:
        {"check": cmd_check, "ask": cmd_ask, "solve": cmd_solve, "ladder": cmd_ladder, "report": cmd_report}[a.cmd](a)
    except LLMError as e:
        print(f"\nОшибка модели: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
