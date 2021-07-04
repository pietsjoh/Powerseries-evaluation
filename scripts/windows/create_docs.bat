@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat

if not exist .\docs\build (mkdir .\docs\build)
if exist .\docs\source (@RD /S /Q ".\docs\source")
mkdir .\docs\source
mkdir .\docs\source\_static
mkdir .\docs\source\_templates
python .\pysrc\setup\create_doc_files.py
cd docs
call make clean
call sphinx-apidoc --module-first --force --no-toc -o source ..\pysrc
call make html
cd ..

call deactivate
cd %current_dir%