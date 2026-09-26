@echo off
rem 一键执行本地体检与全量测试；任何一步失败即停止。
cd /d "%~dp0.."

python scripts\check_project.py
if errorlevel 1 exit /b 1

python -m pytest
if errorlevel 1 exit /b 1

echo.
echo ALL CHECKS PASSED
