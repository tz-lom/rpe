@echo off
pushd "%~dp0\condaenv"
call ".\bin\qt_mingw_32_env.bat
set PATH=%CD%\resonance_mingw_32\bin;%CD%;%CD%\Library\bin;%PATH%
set PYTHONHOME=%CD%
cd  "%~dp0\bin"
start SStimProg %*
popd