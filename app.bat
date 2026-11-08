@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul
python app\server.py
