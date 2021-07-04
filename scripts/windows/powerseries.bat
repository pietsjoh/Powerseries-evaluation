@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
python .\pysrc\powerseries\combine_ps_tool.py
call deactivate
cd %current_dir%
