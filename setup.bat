@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Создаю виртуальное окружение .venv ===
python -m venv .venv || (echo Не найден Python. Установите Python с python.org, галочка "Add python.exe to PATH" & pause & exit /b 1)
call .venv\Scripts\activate.bat
python --version
if not exist .env (copy .env.example .env >nul & echo Создан файл .env - впишите в него ключ модели: OPENAI_API_KEY для модели ОНСОФТ)
echo === Проверяю систему автотестами ===
python -m unittest discover tests
echo.
echo Готово. Дальше: впишите ключ в .env и запустите  run.bat check
pause
