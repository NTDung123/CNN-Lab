@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto check_version
echo Dang tao moi truong Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" goto failed
:check_version
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 goto wrong_version
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Cai dat xong. Mo RUN_DEMO_WINDOWS.bat de chay.
pause
exit /b 0
:wrong_version
echo.
echo Moi truong .venv hien tai khong phai Python 3.12 64-bit hoac khong chay duoc.
echo Hay doi ten thu muc .venv cu, cai Python 3.12 64-bit kem Python Launcher,
echo roi chay lai SETUP_WINDOWS.bat. Chuong trinh khong tu xoa moi truong cu.
pause
exit /b 1
:failed
echo.
echo Cai dat chua thanh cong. Hay cai Python 3.12 64-bit kem Python Launcher va kiem tra Internet.
echo Giu cua so nay de xem thong bao loi o tren.
pause
exit /b 1
