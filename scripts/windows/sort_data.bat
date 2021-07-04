@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
python .\pysrc\data_tools\sort_data.py
call deactivate
cd %current_dir%
