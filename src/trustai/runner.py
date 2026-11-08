"""Прогон набора вопросов, запись answers.csv, журнала и статистики, замер оценщиком."""
import csv
import json
import re
import subprocess
import sys
import time
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
    packages, rows = {}, []
    stats0 = dict(llm.stats)
    t0 = time.time()
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as tf:
        for i, q in enumerate(questions, 1):
            pname = q["package"]
            if pname not in packages:
                ppath = data_dir / "packages" / pname
                try:
                    packages[pname] = load_package(ppath)
                    if packages[pname].warnings:
                        print(f"  [{pname}] предупреждения: {'; '.join(packages[pname].warnings)}")
                except Exception as e:  # закупка не читается: на все её вопросы «не знаю»
                    print(f"  [{pname}] не удалось прочитать: {e}")
                    packages[pname] = None
            pkg = packages[pname]
            if pkg is None:
                res_row = {"id": q["id"], "status": "не знаю", "answer": "", "quote": "", "file": ""}
                rows.append(res_row)
                continue
            try:
                res = solver.solve(q["id"], q["question"], pkg)
            except Exception as e:  # ошибка сети/модели не должна ронять весь прогон
                from .pipeline import Result
                res = Result(q["id"], "не знаю", trace=[{"step": "error", "message": str(e)[:500]}])
                print(f"  {q['id']}: ошибка {e}")
            rows.append(res.row())
            tf.write(json.dumps({"id": q["id"], "question": q["question"], **res.row(), "trace": res.trace},
                                ensure_ascii=False) + "\n")
            if progress:
                print(f"[{i}/{len(questions)}] {q['id']}: {res.status} {res.answer[:60]}")
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
