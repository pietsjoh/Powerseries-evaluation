@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
python .\pysrc\setup\view_docs.py
call deactivate
cd %current_dir%