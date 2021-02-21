if not exist ./condaenv call conda create -q -y --prefix ./condaenv
call conda activate ./condaenv
call conda config --env --set subdir win-32
call conda install -y -c bcilab resonance_mingw_32 resonance_pip
