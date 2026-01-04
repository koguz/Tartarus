@echo off
setlocal enabledelayedexpansion
REM Tartarus D2 Parameter Sweep
REM Population: 512, 1024
REM States: 12, 16, 24, 32, 48, 64, 128
REM Generations: 1000, 2000, 4000, 8000
REM Total: 2 x 7 x 4 = 56 runs

echo ============================================
echo Tartarus D2 Parameter Sweep - 56 configurations
echo Using D2 crossover (gene m_fitness comparison)
echo ============================================
echo.

REM Log file
set LOG=sweep_d2_log.txt

echo Starting D2 sweep at %date% %time% > %LOG%
echo. >> %LOG%

set RUN_ID=1
set COUNT=0

for %%N in (1024) do (
    for %%S in (12 16 24 32 48 64) do (
        for %%K in (1000 2000) do (
            set /a COUNT+=1

            echo ============================================
            echo Run !COUNT!/56: N=%%N, S=%%S, K=%%K [D2]
            echo ============================================
            echo.

            echo [!COUNT!/56] Training D2: N=%%N S=%%S K=%%K >> %LOG%
            echo Start: %time% >> %LOG%

            REM Training with D2
            echo Training with TartarusD2.exe %%N %%S %%K %RUN_ID%
            TartarusD2.exe %%N %%S %%K %RUN_ID%

            echo Training complete. >> %LOG%

            REM Testing
            echo Testing with TartarusTestAll.exe txt/b-D2-%%N-%%S-%%K-%RUN_ID%.txt %%S
            TartarusTestAll.exe txt/b-D2-%%N-%%S-%%K-%RUN_ID%.txt %%S

            echo End: %time% >> %LOG%
            echo. >> %LOG%

            echo.
        )
    )
)

echo ============================================
echo D2 Sweep complete! %COUNT% runs finished.
echo ============================================
echo.
echo Results saved in:
echo   - txt/r-D2-N-S-K-L.txt (training curves with header: generation,best,average)
echo   - txt/b-D2-N-S-K-L.txt (best solutions)
echo   - txt/sc_*.txt (score distributions)
echo   - txt/st_*.txt (state distributions)
echo   - complete_results.csv (summary)
echo   - %LOG% (timing log)

echo. >> %LOG%
echo D2 Sweep complete at %date% %time% >> %LOG%

pause
