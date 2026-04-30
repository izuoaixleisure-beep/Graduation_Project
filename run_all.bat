@echo off
cd /d "%~dp0"
echo ========================================
echo  Puddle Detection - Training All Models
echo ========================================

set PYTHON=.venv\Scripts\python.exe
set DATA=datasets\puddle\data.yaml
set DEVICE=0
set EPOCHS=100

echo.
echo [1/4] Training baseline...
%PYTHON% train_puddle.py --data %DATA% --mode baseline --device %DEVICE% --epochs %EPOCHS%
if %errorlevel% neq 0 ( echo BASELINE FAILED & pause & exit /b 1 )

echo.
echo [2/4] Training ablation_p2...
%PYTHON% train_puddle.py --data %DATA% --mode ablation_p2 --device %DEVICE% --epochs %EPOCHS%
if %errorlevel% neq 0 ( echo ABLATION_P2 FAILED & pause & exit /b 1 )

echo.
echo [3/4] Training ablation_cbam...
%PYTHON% train_puddle.py --data %DATA% --mode ablation_cbam --device %DEVICE% --epochs %EPOCHS%
if %errorlevel% neq 0 ( echo ABLATION_CBAM FAILED & pause & exit /b 1 )

echo.
echo [4/4] Training improved...
%PYTHON% train_puddle.py --data %DATA% --mode improved --device %DEVICE% --epochs %EPOCHS%
if %errorlevel% neq 0 ( echo IMPROVED FAILED & pause & exit /b 1 )

echo.
echo ========================================
echo  All training complete!
echo ========================================
pause
