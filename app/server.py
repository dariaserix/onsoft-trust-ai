"""Веб-интерфейс для живой демонстрации: задать вопрос и увидеть, как система проверяет ответ.

Запуск: app.bat  (или python app/server.py) → откроется http://127.0.0.1:8501
Только стандартная библиотека Python.
"""
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trustai.docs import load_package  # noqa: E402
from trustai.llm import make_llm  # noqa: E402
from trustai.pipeline import Solver  # noqa: E402
from trustai.runner import load_config, read_questions  # noqa: E402

PORT = 8501
LOCK = threading.Lock()
STATE = {}


def datasets():
    out = {}
    for d in sorted((ROOT / "data").iterdir()):
        if (d / "packages").is_dir():
            pk = sorted(p.name for p in (d / "packages").iterdir() if p.is_dir())
            qs = read_questions(d / "questions.csv") if (d / "questions.csv").exists() else []
            if pk:
                out[d.name] = {"packages": pk, "questions": [{"id": q["id"], "package": q["package"],
                                                              "question": q["question"]} for q in qs]}
    return out


def solver():
    if "solver" not in STATE:
        STATE["solver"] = Solver(make_llm(), load_config(ROOT / "configs" / "final.json"))
    return STATE["solver"]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/report":
            p = ROOT / "results" / "report.html"
            if p.exists():
                return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
            return self._send(404, "Дашборда ещё нет: запустите run.bat ladder и run.bat report", "text/plain; charset=utf-8")
        if self.path == "/api/datasets":
            return self._send(200, json.dumps(datasets(), ensure_ascii=False))
        self._send(404, "{}")

    def do_POST(self):
        if self.path != "/api/ask":
            return self._send(404, "{}")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            ds, pkg_name, q = req["dataset"], req["package"], req["question"].strip()
            pkg = load_package(ROOT / "data" / ds / "packages" / pkg_name)
            with LOCK:
                s = solver()
                before = dict(s.llm.stats)
                res = s.solve("ASK", q, pkg)
                calls = s.llm.stats["calls"] - before["calls"]
            self._send(200, json.dumps({"status": res.status, "answer": res.answer, "quote": res.quote, "file": res.file,
                                        "trace": res.trace, "calls": calls, "warnings": pkg.warnings,
                                        "model": s.llm.name}, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Интерфейс: {url}  (остановить: Ctrl+C)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
