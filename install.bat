if not exist ./condaenv call conda create -q -y --prefix ./condaenv
if not exist ./condaenv64 call conda create -q -y --prefix ./condaenv64 
call conda activate ./condaenv
call conda config --env --set subdir win-32
call conda install -y -c bcilab resonance_mingw_32=3.4.1=20 resonance_pip=1.1.2=hfeaa757_10

call conda activate ./condaenv64
call conda config --env --set subdir win-64
call conda install -y -c bcilab resonance_mingw_64=3.4.1=20 resonance_pip=1.1.2=hfeaa757_10