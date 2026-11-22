@echo off
rem RL Labirent kurulumu: .venv olusturur ve paketleri yukler.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Sanal ortam olusturuluyor...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Python bulunamadi. python.org adresinden Python 3.11 veya ustunu kurun.
        pause
        exit /b 1
    )
)
echo Paketler yukleniyor...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Paket kurulumu basarisiz.
    pause
    exit /b 1
)
echo.
echo Kurulum tamam. Uygulamayi baslat.bat ile acabilirsiniz.
pause
