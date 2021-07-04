@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
python .\pysrc\FSR_analysis\fsr_selection.py
call deactivate
cd %current_dir%
