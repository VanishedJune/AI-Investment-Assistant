@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo 正在打开 AI Investment Assistant 四周决策月报...
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" server.py
  goto :done
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" server.py
  goto :done
)
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 server.py
  goto :done
)
where python >nul 2>nul
if %errorlevel%==0 (
  python server.py
  goto :done
)
echo 未找到可用的 Python。请创建项目 .venv 或安装 Python 3.11+。
exit /b 1
:done
pause
endlocal
