@echo off
pushd d:\Projects\BCI_EyeLines_Online_2020\rpe\condaenv
call .\bin\qt_mingw_32_env.bat
set PATH=%CD%\resonance_mingw_32\bin;%CD%;%CD%\Library\bin;%PATH%
set PYTHONHOME=%CD%
cd d:\Projects\BCI_EyeLines_Online_2020\rpe\bin\
