@echo off
set PYTHON_PATH=C:\Users\99san\AppData\Local\Programs\Python\Python310\python.exe

echo ===================================================
echo  Starting Tournament Poker AI Training
echo  Model: deepstack_7actions_dense_v3_ompeval
echo ===================================================
echo.

"%PYTHON_PATH%" train_tournament_dense.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training script failed with error code %errorlevel%
    pause
) else (
    echo.
    echo [SUCCESS] Training completed successfully.
    pause
)
