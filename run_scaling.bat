@echo off

REM === TartarusD2 varying P and L ===
for %%P in (256 512 1024 2048 4096 8192) do (
    for %%L in (3 5 7 9) do (
        TartarusD2 4096 16 3000 %%P %%L
    )
)

REM === TartarusD2 varying S and L ===
for %%S in (1 2 4 8 16 32 64 128) do (
    for %%L in (4 6 8 10) do (
        TartarusD2 4096 %%S 3000 128 %%L
    )
)

REM === TartarusTestAll varying P and L ===
for %%P in (256 512 1024 2048 4096 8192) do (
    for %%L in (3 5 7 9) do (
        TartarusTestAll.exe txt/b-D2-4096-16-3000-%%P-%%L.txt 16
    )
)

REM === TartarusTestAll varying S and L ===
for %%S in (1 2 4 8 16 32 64 128) do (
    for %%L in (4 6 8 10) do (
        TartarusTestAll.exe txt/b-D2-4096-%%S-3000-128-%%L.txt %%S
    )
)
