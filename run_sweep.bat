@echo off
setlocal enabledelayedexpansion
REM Tartarus Parameter Sweep
REM Population: 512, 1024
REM States: 12, 16, 24, 32, 48, 64, 128
REM Generations: 1000, 2000, 4000, 8000
REM Total: 2 x 7 x 4 = 56 runs

echo ============================================
echo Tartarus Parameter Sweep - 56 configurations
echo ============================================
echo.

REM Log file
set LOG=sweep_log.txt

echo Starting sweep at %date% %time% > %LOG%
echo. >> %LOG%

set RUN_ID=1
set COUNT=0

for %%N in (512 1024) do (
    for %%S in (12 16 24 32 48 64 128) do (
        for %%K in (1000 2000 4000 8000) do (
            set /a COUNT+=1

            echo ============================================
            echo Run !COUNT!/56: N=%%N, S=%%S, K=%%K
            echo ============================================
            echo.

            echo [!COUNT!/56] Training: N=%%N S=%%S K=%%K >> %LOG%
            echo Start: %time% >> %LOG%

            REM Training
            echo Training with Tartarus74K.exe %%N %%S %%K %RUN_ID%
            Tartarus74K.exe %%N %%S %%K %RUN_ID%

            echo Training complete. >> %LOG%

            REM Testing
            echo Testing with TartarusTestAll.exe txt/b-all-%%N-%%S-%%K-%RUN_ID%.txt %%S
            TartarusTestAll.exe txt/b-all-%%N-%%S-%%K-%RUN_ID%.txt %%S

            echo End: %time% >> %LOG%
            echo. >> %LOG%

            echo.
        )
    )
)

echo ============================================
echo Sweep complete! %COUNT% runs finished.
echo ============================================
echo.
echo Results saved in:
echo   - txt/r-all-N-S-K-L.txt (training curves with header: generation,best,average)
echo   - txt/b-all-N-S-K-L.txt (best solutions)
echo   - txt/sc_*.txt (score distributions)
echo   - txt/st_*.txt (state distributions)
echo   - complete_results.csv (summary)
echo   - %LOG% (timing log)

echo. >> %LOG%
echo Sweep complete at %date% %time% >> %LOG%

pause
