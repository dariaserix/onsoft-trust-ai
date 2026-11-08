"""Дашборд «было/стало»: один самодостаточный HTML-файл (без интернета и библиотек).

Источник данных: results/ladder.json и папки results/runs/<шаг>/ (answers.csv, result.csv, trace.jsonl, stats.json).
"""
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

E = html.escape

TYPE_NAMES = {
    "факт": "Обычный факт",
    "расчёт": "Расчёт",
    "после изменений": "Значение изменили",
    "расхождение": "Документы противоречат",
    "ловушка: нет в документах": "Ловушка: ответа нет",
    "ловушка: требования нет": "Ловушка: «требования нет»",
    "пустой файл": "Пустой файл",
}
CRITIC_NAMES = {
    "quote": "Критик цитат", "number": "Критик чисел", "calc": "Расчёт кодом", "amendment": "Критик изменений",
    "discrepancy": "Критик расхождений", "yesno": "Критик «да/нет»", "absent": "Перепроверка «нет в документах»",
    "llm_critic": "ИИ-критик", "format": "Формат ответа", "injection": "Защита от инструкций в тексте",
}


def _read_csv(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def stacked_bars(rows, total):
    """Горизонтальные полосы: зелёный — верно, серый — «не знаю», красный — обман."""
    out = []
    for r in rows:
        c, k, l = r.get("correct", 0) or 0, r.get("careful", 0) or 0, r.get("lies", 0) or 0
        idx = r.get("index")
        seg = lambda n, cls, label: (f'<div class="seg {cls}" style="flex:{n}" title="{label}: {n}">'
                                     f'{n if n * 100 / max(total, 1) >= 5 else ""}</div>' if n else "")
        out.append(f'''<div class="bar-row">
  <div class="bar-label"><b>{E(r.get("title", r.get("step", "")))}</b><span>{E(r.get("description", ""))}</span></div>
  <div class="bar">{seg(c, "ok", "Верно")}{seg(k, "unk", "Не знаю")}{seg(l, "lie", "Обман")}</div>
  <div class="bar-idx {"neg" if (idx or 0) < 0 else ""}">{"" if idx is None else f"{idx:g}"}</div>
</div>''')
    return "\n".join(out)


def build_report(results_dir: Path, data_dir: Path) -> Path:
    results_dir = Path(results_dir)
    ladder = json.loads((results_dir / "ladder.json").read_text(encoding="utf-8")) if (results_dir / "ladder.json").exists() else []
    runs = results_dir / "runs"
    if not ladder:  # нет лестницы: собираем из того, что есть
        for d in sorted(runs.glob("*")):
            s = json.loads((d / "score.json").read_text(encoding="utf-8")) if (d / "score.json").exists() else {}
            if s:
                ladder.append({"step": d.name, "title": d.name, **s})
    if not ladder:
        raise SystemExit("Нет результатов. Сначала запустите: run.bat ladder")
    base = ladder[0]
    # итоговая система — прогон configs/final.json (results/runs/final); если его нет — лучший шаг лестницы
    fpath = runs / "final" / "score.json"
    if fpath.exists():
        fst = runs / "final" / "stats.json"
        final = {"step": "final", "title": "Итоговая система", **json.loads(fpath.read_text(encoding="utf-8")),
                 **({"model": json.loads(fst.read_text(encoding="utf-8")).get("model")} if fst.exists() else {})}
    else:
        final = max(reversed(ladder), key=lambda r: r.get("index", -999))
    n = base.get("n") or 1
    gold = {g["id"]: g for g in json.loads((data_dir / "answers_public.json").read_text(encoding="utf-8"))} \
        if (data_dir / "answers_public.json").exists() else {}
    questions = {}
    qcsv = data_dir / "questions.csv"
    if qcsv.exists():
        for r in _read_csv(qcsv):
            questions[r["id"]] = r["question"]

    # --- по типам вопросов: было vs стало
    def by_type(step):
        res = _read_csv(runs / step / "result.csv")
        d = defaultdict(Counter)
        for r in res:
            d[r["type"]][r["verdict"]] += 1
        return d
    bt_base, bt_final = by_type(base["step"]), by_type(final["step"])
    type_rows = []
    for t in sorted(set(bt_base) | set(bt_final), key=lambda t: -sum(bt_final.get(t, Counter()).values())):
        b, f = bt_base.get(t, Counter()), bt_final.get(t, Counter())
        tot = sum(f.values()) or sum(b.values())
        type_rows.append(f'''<tr><td>{E(TYPE_NAMES.get(t, t))}</td><td class="num">{tot}</td>
<td class="num">{b["верно"]} / <span class="lie-t">{b["обман"]}</span></td>
<td class="num">{f["верно"]} / <span class="lie-t">{f["обман"]}</span></td></tr>''')

    # --- типы обмана на старте
    base_res = _read_csv(runs / base["step"] / "result.csv")
    why = Counter(r["why"] for r in base_res if r["verdict"] == "обман")
    why_rows = "".join(f'<tr><td>{E(w)}</td><td class="num">{c}</td></tr>' for w, c in why.most_common())

    # --- вклад каждого шага
    contrib = []
    for prev, cur in zip(ladder, ladder[1:]):
        d = (cur.get("index") or 0) - (prev.get("index") or 0)
        dl = (cur.get("lies") or 0) - (prev.get("lies") or 0)
        contrib.append(f'<tr><td>{E(cur.get("title", cur["step"]))}</td><td class="num {"pos" if d > 0 else "neg" if d < 0 else ""}">'
                       f'{d:+.1f}</td><td class="num">{dl:+d}</td></tr>')

    # --- пойманные обманы (из трасс итоговой системы на всех наборах, где есть правильные ответы)
    def load_gold(d):
        p = Path(d) / "answers_public.json"
        return {g["id"]: g for g in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}

    root = Path(data_dir).parent
    sources = [("учебный набор", final["step"], base["step"], gold)]
    for label, fin, bas, d in (("переформулированные вопросы", "para_final", "para_00_baseline", "paraphrase"),
                               ("трудные файлы", "hard_final", "hard_00_baseline", "hard"),
                               ("новые типы вопросов", "newtypes_final", "newtypes_00_baseline", "newtypes"),
                               ("без рассуждений, учебный набор", "nothink_train_final", "nothink_train_00_baseline", "train"),
                               ("без рассуждений, другие формулировки", "nothink_paraphrase_final",
                                "nothink_paraphrase_00_baseline", "paraphrase")):
        if (runs / fin / "trace.jsonl").exists():
            sources.append((label, fin, bas, load_gold(root / d)))
    examples = []
    for label, fin, bas, gd in sources:
        fres = {r["id"]: r for r in _read_csv(runs / fin / "result.csv")} if (runs / fin / "result.csv").exists() else {}
        bans = {r["id"]: r for r in _read_csv(runs / bas / "answers.csv")} if (runs / bas / "answers.csv").exists() else {}
        for t in _read_jsonl(runs / fin / "trace.jsonl"):
            evs = [e for s in t["trace"] for e in s.get("events", []) if e.get("caught")]
            evs += [s for s in t["trace"] if s.get("step") == "injection"][:1]
            if not evs:
                continue
            ok = fres.get(t["id"], {}).get("verdict")
            first = next((s.get("first_answer") for s in t["trace"] if s.get("step") == "final"), None) or {}
            # сначала — где первый ответ модели был другим (обман пойман и исправлен), потом остальные
            changed = str(first.get("answer", "")) != str(t.get("answer", "")) or first.get("status") != t.get("status")
            kinds = tuple(sorted(set(e.get("critic", "") for e in evs)))
            examples.append(((0 if ok == "верно" else 1, 0 if changed else 1), kinds, label, t, evs, ok, first,
                             gd.get(t["id"], {}), bans.get(t["id"], {})))
    examples.sort(key=lambda x: x[0])
    picked, seen_kinds = [], set()
    for ex in examples:  # по одному примеру на каждый вид пойманного обмана, затем остальные
        if ex[1] not in seen_kinds:
            seen_kinds.add(ex[1])
            picked.append(ex)
    picked += [ex for ex in examples if ex not in picked and ex[1] != ("injection",)]
    ex_html = []
    for _, _, label, t, evs, ok, first, g, b in picked[:10]:
        ev_html = "".join(f'<li><b>{E(CRITIC_NAMES.get(e.get("critic"), e.get("critic", "")))}:</b> {E(e.get("message", ""))}</li>' for e in evs[:3])
        ex_html.append(f'''<article class="case">
  <div class="case-q"><span class="pill">{E(t["id"])}</span> {E(t.get("question", ""))} <span class="src">{E(label)}</span></div>
  <div class="case-grid">
    <div><h4>Было (модель без проверок)</h4><p>{E(b.get("status", "—"))}: <b>{E(b.get("answer", "") or "—")}</b></p></div>
    <div><h4>Модель сначала ответила</h4><p>{E(str(first.get("status", "—")))}: <b>{E(str(first.get("answer", "") or "—"))}</b></p></div>
    <div><h4>Итог системы</h4><p>{E(t.get("status", ""))}: <b>{E(t.get("answer", "") or "—")}</b>
      <span class="verdict {"v-ok" if ok == "верно" else "v-unk" if ok == "осторожно" else "v-lie"}">{E(ok or "")}</span></p></div>
  </div>
  <ul class="caught">{ev_html}</ul>
  {f'<p class="truth">Правильно: {E(g.get("status", ""))} {E(" / ".join(map(str, g.get("expected", []))))}</p>' if g else ""}
</article>''')

    def sc(step):
        p = runs / step / "score.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    # --- режим модели: с рассуждениями и без (одна и та же модель, меняется только режим)
    def stat(step):
        p = runs / step / "stats.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    mode_rows = []
    for label, pre in (("с рассуждениями (итоговый режим)", ""), ("без рассуждений", "nothink_")):
        cells = ""
        for bas, fin in ((f"{pre}train_00_baseline" if pre else base["step"], f"{pre}train_final" if pre else final["step"]),
                         (f"{pre}paraphrase_00_baseline" if pre else "para_00_baseline",
                          f"{pre}paraphrase_final" if pre else "para_final")):
            b0, f0 = sc(bas), sc(fin)
            cells += (f'<td class="num">{b0.get("index")} → <b>{f0.get("index")}</b></td>' if b0 and f0 else "<td></td>")
        st = stat(f"{pre}train_final" if pre else final["step"])
        q = st.get("questions") or n
        sec = st.get("model_seconds") or 0
        cells += f'<td class="num">{sec / q:.1f}</td>' if sec else "<td></td>"
        mode_rows.append(f"<tr><td>{E(label)}</td>{cells}</tr>")

    # --- проверка на данных, на которых систему не настраивали
    gen_rows = []
    for label, note, bas, fin in (
            ("Учебный набор организатора", "на нём разбирали ошибки", base["step"], final["step"]),
            ("Те же закупки, вопросы другими словами", "71 вопрос переформулирован вручную", "para_00_baseline", "para_final"),
            ("Трудные файлы", "кодировка Windows-1251, пустой файл, спрятанная инструкция для ИИ, документ раздут в 15 раз", "hard_00_baseline", "hard_final"),
            ("Новые типы вопросов", "24 вопроса о том, о чём учебный набор не спрашивает (заказчик, пени, таблицы, числа "
             "словами, новые расчёты и ловушки); ответы выписаны вручную", "newtypes_00_baseline", "newtypes_final")):
        b0, f0 = sc(bas), sc(fin)
        if b0 and f0:
            gen_rows.append(f'''<tr><td><b>{E(label)}</b><br><span class="note">{E(note)}</span></td><td class="num">{f0.get("n")}</td>
<td class="num">{b0.get("index")} <span class="lie-t">({b0.get("lies")} обм.)</span></td>
<td class="num"><b>{f0.get("index")}</b> <span class="lie-t">({f0.get("lies")} обм.)</span></td></tr>''')

    # --- вклад каждой проверки: выключаем по одной из итоговой системы
    loo = {}
    for tag in ("train", "paraphrase"):
        p = results_dir / f"loo_{tag}.json"
        if p.exists():
            loo[tag] = json.loads(p.read_text(encoding="utf-8"))
    loo_rows = []
    if loo:
        tags = list(loo)
        full = {t: loo[t][0] for t in tags}
        titles = [r["title"] for r in loo[tags[0]]][1:]
        for title in titles:
            cells = ""
            for t in tags:
                r = next((x for x in loo[t] if x["title"] == title), None)
                if not r:
                    cells += "<td></td>"
                    continue
                d = (r.get("index") or 0) - (full[t].get("index") or 0)
                dl = (r.get("lies") or 0) - (full[t].get("lies") or 0)
                cls = "neg" if d < 0 else "pos" if d > 0 else ""
                cells += (f'<td class="num"><span class="{cls}">{d:+.1f}</span>'
                          f'{f" <span class=lie-t>(+{dl} обм.)</span>" if dl > 0 else ""}</td>')
            loo_rows.append(f"<tr><td>без: {E(title)}</td>{cells}</tr>")
        loo_head = "".join(f'<th class="num">{"учебный набор" if t == "train" else "другие формулировки"}</th>' for t in tags)
    else:
        loo_head = ""

    fs = json.loads((runs / final["step"] / "stats.json").read_text(encoding="utf-8")) if (runs / final["step"] / "stats.json").exists() else {}

    def kpi(label, a, b, good_up=True, suffix=""):
        d = (b or 0) - (a or 0)
        cls = "pos" if (d > 0) == good_up and d != 0 else "neg" if d else ""
        return (f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-val">{b:g}{suffix}</div>'
                f'<div class="kpi-delta {cls}">было {a:g}{suffix} → {d:+g}</div></div>')

    html_doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ИИ, который не обманывает</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#14171f; --muted:#5d6472; --line:#e3e6ec;
  --ok:#1f9d63; --unk:#b7bdc8; --lie:#d9443b; --accent:#2f5bea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1115; --card:#181b22; --ink:#e9ecf2; --muted:#9aa2b1;
  --line:#2a2f3a; --unk:#4a5160; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif }}
main {{ max-width:1080px; margin:0 auto; padding:32px 16px 64px }}
h1 {{ font-size:32px; margin:0 0 4px }} h2 {{ font-size:22px; margin:40px 0 12px }} h4 {{ margin:0 0 4px; font-size:13px; color:var(--muted); font-weight:600 }}
.sub {{ color:var(--muted); margin:0 0 24px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px }}
.kpi-label {{ color:var(--muted); font-size:14px }} .kpi-val {{ font-size:40px; font-weight:700; line-height:1.1 }}
.kpi-delta {{ font-size:14px; color:var(--muted) }} .pos {{ color:var(--ok) }} .neg {{ color:var(--lie) }}
.explain {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-top:12px }}
.explain div {{ font-size:14px; color:var(--muted) }} .explain b {{ color:var(--ink) }}
.legend {{ display:flex; gap:16px; font-size:14px; color:var(--muted); margin-bottom:12px; flex-wrap:wrap }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px }}
.bar-row {{ display:grid; grid-template-columns:minmax(160px,300px) 1fr 52px; gap:12px; align-items:center; padding:8px 0; border-top:1px solid var(--line) }}
.bar-label b {{ display:block; font-size:15px }} .bar-label span {{ display:block; font-size:12px; color:var(--muted) }}
.bar {{ display:flex; height:26px; border-radius:6px; overflow:hidden; background:var(--line) }}
.seg {{ display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:600 }}
.ok {{ background:var(--ok) }} .unk {{ background:var(--unk) }} .lie {{ background:var(--lie) }}
.bar-idx {{ font-weight:700; text-align:right; font-variant-numeric:tabular-nums }}
table {{ width:100%; border-collapse:collapse; font-size:15px }} th,td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line) }}
th {{ color:var(--muted); font-weight:600; font-size:13px }} .num {{ text-align:right; font-variant-numeric:tabular-nums }}
.lie-t {{ color:var(--lie) }} .note {{ color:var(--muted); font-size:13px }}
.src {{ color:var(--muted); font-size:12px; font-weight:400; margin-left:6px }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px }}
.case {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; margin-bottom:12px }}
.case-q {{ font-weight:600; margin-bottom:10px }} .pill {{ background:var(--accent); color:#fff; border-radius:6px; padding:1px 7px; font-size:12px; margin-right:6px }}
.case-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px }} .case-grid p {{ margin:0 }}
.caught {{ margin:12px 0 0; padding-left:18px; font-size:14px }} .truth {{ font-size:13px; color:var(--muted); margin:8px 0 0 }}
.verdict {{ font-size:12px; border-radius:6px; padding:1px 6px; margin-left:6px; color:#fff }}
.v-ok {{ background:var(--ok) }} .v-unk {{ background:var(--unk) }} .v-lie {{ background:var(--lie) }}
footer {{ color:var(--muted); font-size:13px; margin-top:40px }}
@media (max-width:640px) {{ .bar-row {{ grid-template-columns:1fr 44px }} .bar-label {{ grid-column:1/-1 }} }}
</style></head><body><main>
<h1>ИИ, который не обманывает</h1>
<p class="sub">Ответы на вопросы по документам госзакупок. Модель: {E(str(final.get("model", fs.get("model", ""))))}. Вопросов: {n}.</p>

<section class="kpis">
  <div class="card">{kpi("Индекс доверия", base.get("index", 0), final.get("index", 0))}</div>
  <div class="card">{kpi("Верных ответов", base.get("correct", 0), final.get("correct", 0))}</div>
  <div class="card">{kpi("Обманов", base.get("lies", 0), final.get("lies", 0), good_up=False)}</div>
  <div class="card">{kpi("Честных «не знаю»", base.get("careful", 0), final.get("careful", 0), good_up=True)}</div>
</section>
<div class="card explain">
  <div><b>Индекс доверия</b> = (верные ответы − обманы) / все вопросы × 100. Максимум 100.</div>
  <div><b>Обман</b> — неверное число, выдуманная цитата, ответ на то, чего нет в документах.</div>
  <div><b>«Не знаю»</b> не добавляет и не отнимает баллы: честное «не знаю» лучше уверенной ошибки.</div>
</div>

<h2>Как каждый шаг уменьшал обман</h2>
<div class="card">
  <div class="legend"><span><i class="dot ok"></i>верно</span><span><i class="dot unk"></i>«не знаю»</span><span><i class="dot lie"></i>обман</span><span>справа — индекс доверия</span></div>
  {stacked_bars(ladder, n)}
</div>

<div class="two">
  <div><h2>Разбивка по типам вопросов</h2><div class="card"><table>
    <tr><th>Тип вопроса</th><th class="num">Всего</th><th class="num">Было: верно / обман</th><th class="num">Стало: верно / обман</th></tr>
    {"".join(type_rows)}</table></div></div>
  <div><h2>Как обманывала модель «как есть»</h2><div class="card"><table>
    <tr><th>Вид обмана</th><th class="num">Случаев</th></tr>{why_rows or '<tr><td colspan=2>нет обманов</td></tr>'}</table></div>
    <h2>Вклад каждого шага</h2><div class="card"><table>
    <tr><th>Шаг</th><th class="num">Индекс</th><th class="num">Обманов</th></tr>{"".join(contrib)}</table></div></div>
</div>

<h2>Режим модели: точность и скорость</h2>
<p class="sub">Та же модель, меняется только режим «рассуждать перед ответом». Индекс: модель как есть → наша система.</p>
<div class="card"><table>
  <tr><th>Режим</th><th class="num">Учебный набор</th><th class="num">Другие формулировки</th><th class="num">Секунд модели на вопрос</th></tr>
  {"".join(mode_rows)}
</table></div>

<h2>Работает ли на том, на чём её не настраивали</h2>
<div class="card"><table>
  <tr><th>Набор</th><th class="num">Вопросов</th><th class="num">Было: индекс</th><th class="num">Стало: индекс</th></tr>
  {"".join(gen_rows) or '<tr><td colspan=4>нет данных</td></tr>'}
</table></div>

{f'''<h2>Вклад каждой проверки</h2>
<p class="sub">Выключаем одну проверку в итоговой системе и смотрим, насколько упадёт индекс доверия. Ноль — на этих вопросах проверка не понадобилась.</p>
<div class="card"><table><tr><th>Что выключили</th>{loo_head}</tr>{"".join(loo_rows)}</table></div>''' if loo_rows else ""}

<h2>Обманы, которые система поймала</h2>
{"".join(ex_html) or '<p class="sub">Пока нет примеров.</p>'}

<h2>Стоимость</h2>
<div class="card"><table>
  <tr><th>Шаг</th><th class="num">Запросов к модели</th><th class="num">Токенов</th><th class="num">Время ответов модели, с</th></tr>
  {"".join(f'<tr><td>{E(r.get("title", r["step"]))}</td><td class="num">{r.get("calls", "")}</td><td class="num">{r.get("tokens", "")}</td><td class="num">{round(r.get("model_seconds") or r.get("seconds") or 0)}</td></tr>' for r in ladder)}
</table></div>
<footer>Замер официальным check.py организатора на одной и той же модели. Система не видит правильных ответов: их использует только оценщик.</footer>
</main></body></html>"""
    out = results_dir / "report.html"
    out.write_text(html_doc, encoding="utf-8")
    return out
