@echo off
set current_dir=%cd%
cd %~dp0
cd ..\..
call .\env\Scripts\activate.bat
python .\pysrc\FSR_analysis\plot_fsr.py
call deactivate
cd %current_dir%
