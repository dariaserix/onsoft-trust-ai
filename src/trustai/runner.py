"""Прогон набора вопросов, запись answers.csv, журнала и статистики, замер оценщиком."""
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .docs import load_package
from .llm import ROOT, BaseLLM
from .pipeline import Solver


def read_questions(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")
    first = text.split("\n", 1)[0]
    delim = ";" if first.count(";") >= first.count(",") else ","
    rows = []
    for r in csv.DictReader(text.splitlines(), delimiter=delim):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if r.get("id"):
            if not r.get("package"):
                r["package"] = r["id"].split("-")[0]
            rows.append(r)
    return rows


def load_config(path):
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def run_set(llm: BaseLLM, cfg: dict, data_dir, out_dir, only=None, limit=None, progress=True):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = read_questions(data_dir / "questions.csv")
    if only:
        questions = [q for q in questions if q["package"] in only or q["id"] in only]
    if limit:
        questions = questions[:int(limit)]
    solver = Solver(llm, cfg)
    packages = {}
    stats0 = dict(llm.stats)
    t0 = time.time()
    for q in questions:  # документы читаем заранее, по одному разу на закупку
        pname = q["package"]
        if pname in packages:
            continue
        try:
            packages[pname] = load_package(data_dir / "packages" / pname)
            if packages[pname].warnings:
                print(f"  [{pname}] предупреждения: {'; '.join(packages[pname].warnings)}")
            solver.retriever(packages[pname])
        except Exception as e:  # закупка не читается: на все её вопросы «не знаю»
            print(f"  [{pname}] не удалось прочитать: {e}")
            packages[pname] = None

    def work(q):
        pkg = packages[q["package"]]
        if pkg is None:
            from .pipeline import Result
            return Result(q["id"], "не знаю", trace=[{"step": "unreadable", "message": "закупка не читается"}])
        try:
            return solver.solve(q["id"], q["question"], pkg)
        except Exception as e:  # ошибка сети/модели не должна ронять весь прогон
            from .pipeline import Result
            print(f"  {q['id']}: ошибка {e}")
            return Result(q["id"], "не знаю", trace=[{"step": "error", "message": str(e)[:500]}])

    # Ключ ОНСОФТ разрешает 2 параллельных запроса; LLM_WORKERS=1 — строго по очереди.
    workers = max(1, int(os.environ.get("LLM_WORKERS", "2")))
    results = [None] * len(questions)
    todo = list(range(len(questions)))
    # Сбой сети ≠ «не знаю по документам»: такие вопросы переспрашиваем после паузы.
    # Уже полученные ответы модели берутся из кэша, поэтому повтор ничего не меняет в честности замера.
    for rnd in range(1 + int(os.environ.get("NET_RETRY_ROUNDS", "3"))):
        if rnd:
            pause = 30 * rnd
            print(f"  Сбой связи с моделью на {len(todo)} вопросах. Повтор через {pause} с (круг {rnd})...", flush=True)
            time.sleep(pause)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, questions[i]): i for i in todo}
            done = 0
            for f in as_completed(futs):
                i = futs[f]
                results[i] = f.result()
                done += 1
                if progress:
                    print(f"[{done}/{len(futs)}] {questions[i]['id']}: {results[i].status} {results[i].answer[:60]}",
                          flush=True)
        todo = [i for i in todo if any(st.get("step") == "error" for st in results[i].trace)]
        if not todo:
            break
    if todo:
        print(f"  ВНИМАНИЕ: из-за сбоя связи {len(todo)} вопросов остались «не знаю»: "
              + ", ".join(questions[i]["id"] for i in todo))
    rows = [r.row() for r in results]
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as tf:
        for q, res in zip(questions, results):
            tf.write(json.dumps({"id": q["id"], "question": q["question"], **res.row(), "trace": res.trace},
                                ensure_ascii=False) + "\n")
    write_answers(rows, out_dir / "answers.csv")
    st = {k: llm.stats[k] - stats0.get(k, 0) for k in llm.stats}
    st.update({"questions": len(questions), "wall_seconds": round(time.time() - t0, 1), "model": llm.name,
               "config": cfg.get("name", ""), "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))})
    (out_dir / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir / "answers.csv", st


def write_answers(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "answer", "quote", "file"], delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "").replace("\n", " ") for k in w.fieldnames})


def score(answers_csv, data_dir, out_csv=None):
    """Запускает официальный check.py. Возвращает dict с итогом или None, если ответов нет."""
    data_dir = Path(data_dir)
    gold = data_dir / "answers_public.json"
    if not gold.exists():
        return None
    checker = data_dir / "check.py"
    if not checker.exists():
        checker = ROOT / "data" / "train" / "check.py"
    cmd = [sys.executable, str(checker), str(answers_csv), "--gold", str(gold), "--packages", str(data_dir / "packages")]
    if out_csv:
        cmd += ["--out", str(out_csv)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"RESULT (\{.*\})", p.stdout)
    res = json.loads(m.group(1)) if m else None
    return {"summary": res, "text": p.stdout}
