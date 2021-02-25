if not exist ./condaenv call conda create -q -y --prefix ./condaenv
call conda activate ./condaenv
call conda config --env --set subdir win-32
call conda install -y -c bcilab resonance_mingw_32=3.3.1=84 resonance_pip=1.0.1=py38h5f401a8_0
