@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Hay chay SETUP_WINDOWS.bat truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 (
  echo Moi truong Python khong hop le. Hay chay SETUP_WINDOWS.bat de xem huong dan.
  pause
  exit /b 1
)
echo Mo trinh duyet tai: http://127.0.0.1:5000
echo Neu truyen --port, dung dia chi Running on hien ben duoi.
echo Giu cua so nay trong khi su dung. Nhan Ctrl+C de dung.
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 (
  pause
  exit /b 1
)
exit /b 0
