@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Открываю временную ссылку на демо в интернете.
echo Сначала должно быть запущено демо: app.bat. Не закрывайте это окно, пока нужна ссылка.
echo Ссылка - строка вида https://....lhr.life (каждый запуск даёт новую ссылку).
echo.
ssh -tt -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -R 80:127.0.0.1:8501 nokey@localhost.run
pause
