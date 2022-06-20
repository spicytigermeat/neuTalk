@ECHO OFF
set CONDAPATH=miniconda
set ENVPATH=%CONDAPATH%
set NTKPTH=neutalk.pyw

call %CONDAPATH%/Scripts/activate.bat %EVNPATH%

python %NTKPTH%

pause