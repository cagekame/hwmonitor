@echo off
setlocal

REM Percorsi relativi alla cartella di questo .bat
set "ICON=%~dp0hwmonitor.ico"
set "MAIN=%~dp0main.py"
set "NAME=HWmonitor"

REM Costruisci l'exe
python -m PyInstaller ^
  --onefile ^
  --windowed ^
  --name "%NAME%" ^
  --icon "%ICON%" ^
  --add-data "%ICON%;." ^
  "%MAIN%"

echo.
echo Fatto. Troverai l'eseguibile in .\dist\%NAME%.exe
pause
