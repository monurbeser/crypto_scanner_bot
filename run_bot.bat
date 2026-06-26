@echo off
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d C:\bots\crypto_scanner_bot

:START

echo ==================================================== >> bot.log
echo Restart %date% %time% >> bot.log

call .venv\Scripts\activate.bat

python -u main.py >> bot.log 2>&1

echo Python crashed. Restarting in 15 seconds... >> bot.log

timeout /t 15 > nul

goto START