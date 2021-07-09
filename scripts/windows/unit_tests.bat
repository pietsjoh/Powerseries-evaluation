@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
python -m pytest .\tests\unit_tests -xvs --repeat %1
call deactivate
cd %current_dir%
