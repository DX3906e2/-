@echo off
setlocal
rem %~dp0 = 本 bat 所在目录（双击时保证工作目录正确, 否则 import config 会失败）
set "ROOT=%~dp0"
rem 兜底: bat 被复制到别处(如桌面)时, 回退到项目根目录
if not exist "%ROOT%src\app.py" set "ROOT=C:\Users\ZhuanZ\Desktop\神经网络\"
cd /d "%ROOT%"
if not exist "src\app.py" (
  echo [ERROR] 找不到 src\app.py, 请把本文件放回项目根目录
  pause
  exit /b 1
)
set "PYW=C:\PROGRA~2\MICROS~2\Shared\PYTHON~1\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" src\app.py
