set PYTHONPATH=%~dp0
cd ./condaenv64/bin
call ./SPEED_mingw_64.bat --serviceConfig "%~dp0\speed.json"
