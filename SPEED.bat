set PYTHONPATH=%~dp0
cd ./condaenv/bin
call ./SPEED_mingw_32.bat --serviceConfig "%~dp0\speed.json"
