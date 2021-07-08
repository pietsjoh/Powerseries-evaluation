@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
mypy .\pysrc\utils
call deactivate
cd %current_dir%